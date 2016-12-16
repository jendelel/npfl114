#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import sys
import tools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

import nli_dataset

class Network:
    LANGUAGES = 11

    def _max_pool(self, inp, kernel_size, stride):
        with self.session.graph.as_default():
            mp_pre = tf.expand_dims(inp, axis=2)
            mp = tf_layers.max_pool2d(inputs=mp_pre, kernel_size=kernel_size, stride=stride)
            mp_post = tf.squeeze(mp, axis=2)
            print("mp", mp_post.get_shape())
            return mp_post

    def _1d_conv(self, inp, num_outputs, kernel_size=3, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm):
        with self.session.graph.as_default():
            conv_1 = tf_layers.convolution2d(inputs=inp, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride,
                                             activation_fn=activation_fn, normalizer_fn=normalizer_fn, padding='VALID')

            #if stride == 1:
            #    conv_2 += tf_layers.linear(inputs=inp, num_outputs=num_outputs)
            #else:
            #    conv_2 += tf_layers.convolution2d(inputs=inp, num_outputs=num_outputs, kernel_size=1, stride=stride, normalizer_fn=None, activation_fn=None)  # residual connections
            #print("Conv:", conv_2.get_shape())
            return conv_1

    def __init__(self, rnn_cell, rnn_cell_dim, num_words, num_chars, logdir, expname, threads=1, seed=42, word_embedding=100, char_embedding=100, keep_prob=0.5, num_filters=512, l2=0.001, pretrained=None):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.essay_lens = tf.placeholder(tf.int32, [None])
            self.sentence_ids = tf.placeholder(tf.int32, [None, None])
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.charseq_ids = tf.placeholder(tf.int32, [None, None])
            self.charseqs = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.languages = tf.placeholder(tf.int32, [None])
            self.is_training = tf.placeholder_with_default(False, [])

            input_chars = tf.nn.embedding_lookup(tf.get_variable("char_emb", shape=[num_chars, 25]),
                                                 self.charseqs)
            print("input_chars", input_chars.get_shape())

            if rnn_cell == "LSTM":
                rnn_cell_ce = tf.nn.rnn_cell.LSTMCell(25)
                rnn_cell_50 = tf.nn.rnn_cell.LSTMCell(50)
                rnn_cell_100 = tf.nn.rnn_cell.LSTMCell(100)
            elif rnn_cell == "GRU":
                rnn_cell_25 = tf.nn.rnn_cell.GRUCell(25)
                rnn_cell_50 = tf.nn.rnn_cell.GRUCell(50)
                rnn_cell_100 = tf.nn.rnn_cell.GRUCell(100)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_25, rnn_cell_25, input_chars,
                                                                          self.charseq_lens, dtype=tf.float32, scope="rnn_chars")
            input_chars = tf.concat(1, [state_fw, state_bw])
            print("input_chars, c", input_chars.get_shape())

            input_char_words = tf.nn.embedding_lookup(input_chars, self.charseq_ids)
            print("input_char_words", input_char_words.get_shape())

            word_emb = tf.get_variable("word_emb", shape=[num_words, 300], initializer=tf.constant_initializer(pretrained))
            #word_emb = tf.get_variable("word_emb", shape=[num_words, 300])
            input_words = tf.nn.embedding_lookup(word_emb, self.word_ids)
            print("input_words, t", input_words.get_shape())
            inputs = tf.concat(2, [input_char_words, input_words])
            print("inputs, e_i", inputs.get_shape())

            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_100, rnn_cell_100, inputs,
                                                                      self.sentence_lens, dtype=tf.float32,
                                                                      scope="rnn_words")

            states_concat = tf.concat(1, [state_fw, state_bw])
            print("states_concat_word, s", states_concat.get_shape())

            sentence_embeddings = tf.nn.embedding_lookup(states_concat, self.sentence_ids)
            print("sentence_embeddings, s", sentence_embeddings.get_shape())

            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_100, rnn_cell_100, sentence_embeddings,
                                                                      self.essay_lens, dtype=tf.float32,
                                                                      scope="rnn_sentences")

            states_concat = tf.concat(1, [state_fw, state_bw])
            print("states_concat_sentence", states_concat.get_shape())

            output_layer = tf_layers.linear(states_concat, num_outputs=self.LANGUAGES)
            self.loss = loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.languages)
            self.training = tf.train.AdamOptimizer(1e-4).minimize(loss, self.global_step)
            self.predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.languages)

            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.merge_summary([tf.scalar_summary(self.dataset_name+"/loss", loss),
                                             tf.scalar_summary(self.dataset_name+"/accuracy", self.accuracy)])

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages):
        _, summary = \
            self.session.run([self.training, self.summary],
                             {self.essay_lens: essay_lens, self.sentence_ids: sentence_ids,
                              self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: "train", self.is_training: True})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, dataset):
        accuracy, loss, summary = \
            self.session.run([self.accuracy, self.loss, self.summary],
                             {self.essay_lens: essay_lens, self.sentence_ids: sentence_ids,
                              self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: dataset})
        self.summary_writer.add_summary(summary, self.training_step)
        return (accuracy, loss)

    def predict(self, essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens):
        return self.session.run(self.predictions,
                                {self.essay_lens: essay_lens, self.sentence_ids: sentence_ids,
                                 self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                                 self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="nli-dataset/nli-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="nli-dataset/nli-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="nli-dataset/nli-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=200, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--word_embedding", default=300, type=int, help="word_embedding")
    parser.add_argument("--char_embedding", default=30, type=int, help="char_embedding")
    parser.add_argument("--keep_prob", default=0.5, type=float, help="dropout probability")
    parser.add_argument("--num_filters", default=512, type=int, help="number of output filters from convolution")
    parser.add_argument("--l2", default=0.001, type=float, help="l2 lambda")

    args = parser.parse_args()

    # Load word embeddings
    print("Loading pre trained embeddings.", file=sys.stderr)
    import pandas
    import csv
    with open("nli-dataset/glove.6B.300d.txt", "r") as f:
        w_emb = pandas.read_csv(f, quoting=csv.QUOTE_NONE, delimiter=" ").as_matrix()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = nli_dataset.NLIDataset(args.data_train, pretrained=w_emb)
    data_dev = nli_dataset.NLIDataset(args.data_dev, train=data_train)
    data_test = nli_dataset.NLIDataset(args.data_test, train=data_train, no_languages=True)

    w_emb[:, 0] = np.vectorize(lambda x: data_train.vocabulary_map("words")[x])(w_emb[:, 0])

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "{}-{}{}-bs{}-epochs{}-char{}-word{}-nf{}-l2:{}".format(tools.exp_name(__file__), args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs, args.char_embedding, args.word_embedding, args.num_filters, args.l2)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim,
                      num_words=len(data_train.vocabulary('words')), num_chars=len(data_train.vocabulary('chars')),
                      logdir=args.logdir, expname=expname, threads=args.threads,
                      word_embedding=args.word_embedding, char_embedding=args.char_embedding,
                      keep_prob=args.keep_prob, num_filters=args.num_filters,
                      l2=args.l2, pretrained=w_emb)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_train.next_batch(args.batch_size)
            network.train(essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages)

        essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
            data_dev.whole_data_as_batch()
        dev_accuracy, dev_loss = network.evaluate(essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, "dev")
        print("Development accuracy after epoch {} is {:.2f}. Dev loss is {:.2f}".format(epoch + 1, 100. * dev_accuracy, dev_loss), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_test.whole_data_as_batch()
            test_predictions = network.predict(essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens)

    # Print test predictions
    for prediction in test_predictions:
        print(data_test.vocabulary('languages')[prediction])
