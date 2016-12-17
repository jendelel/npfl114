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

    def __init__(self, rnn_cell, rnn_cell_dim, num_words, num_chars, logdir, expname, threads=1, seed=42, word_embedding=100, char_embedding=100, keep_prob=0.5, num_filters=512, l2=0.001):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            with tf.variable_scope("l2", regularizer=tf_layers.l2_regularizer(l2)):
                if rnn_cell == "LSTM":
                    rnn_cell_co = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
                elif rnn_cell == "GRU":
                    rnn_cell_co = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
                else:
                    raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

                self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
                self.sentence_lens = tf.placeholder(tf.int32, [None])
                self.word_ids = tf.placeholder(tf.int32, [None, None])
                self.charseq_ids = tf.placeholder(tf.int32, [None, None])
                self.charseqs = tf.placeholder(tf.int32, [None, None])
                self.charseq_lens = tf.placeholder(tf.int32, [None])
                self.languages = tf.placeholder(tf.int32, [None])
                self.is_training = tf.placeholder_with_default(False, [])
                self.keep_prob = tf.placeholder_with_default(1.0, [])

                rnn_cell_co = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_co, self.keep_prob, self.keep_prob)

                if char_embedding == -1:
                    input_chars = tf.one_hot(self.charseqs, num_chars)
                else:
                    input_chars = tf.nn.embedding_lookup(tf.get_variable("char_emb", shape=[num_chars, char_embedding]),
                                                         self.charseqs)
                print("input_chars", input_chars.get_shape())

                if rnn_cell == "LSTM":
                    rnn_cell_ce = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
                elif rnn_cell == "GRU":
                    rnn_cell_ce = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
                else:
                    raise ValueError("Unknown rnn_cell {}".format(rnn_cell))
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_ce, rnn_cell_ce, input_chars,
                                                                              self.charseq_lens, dtype=tf.float32, scope="rnn_chars")
                input_chars = state_fw + state_bw
                print("input_chars", input_chars.get_shape())

                input_char_words = tf.nn.embedding_lookup(input_chars, self.charseq_ids)
                print("input_char_words", input_char_words.get_shape())

                if word_embedding == -1:
                    input_words = tf.one_hot(self.word_ids, num_words)
                else:
                    input_words = tf.nn.embedding_lookup(tf.get_variable("word_emb", shape=[num_words, word_embedding]),
                                                         self.word_ids)
                print("input_words", input_words.get_shape())
                inputs = tf.concat(2, [input_char_words, input_words])
                print("inputs", inputs.get_shape())

                (context_fw, context_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_co, rnn_cell_co, inputs, self.sentence_lens, dtype=tf.float32)
                print("state_fw", state_fw.get_shape())


                x = tf.concat(2, [context_fw, inputs, context_bw])
                print("x", x.get_shape())
                y = tf_layers.linear(x, num_outputs=num_filters, activation_fn=tf.tanh)
                print("y", x.get_shape())

                mp = tf.reduce_max(y, axis=1)
                print("mp", mp.get_shape())

                d1 = tf_layers.dropout(mp, keep_prob=self.keep_prob)
                hidden = tf_layers.linear(inputs=d1, num_outputs=300)
                d2 = tf_layers.dropout(hidden, keep_prob=self.keep_prob)
                output_layer = tf_layers.linear(inputs=d2, num_outputs=self.LANGUAGES)

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

    def train(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, keep_prob):
        _, summary = \
            self.session.run([self.training, self.summary],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: "train", self.is_training: True, self.keep_prob:keep_prob})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, dataset):
        accuracy, loss, summary = \
            self.session.run([self.accuracy, self.loss, self.summary],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: dataset})
        self.summary_writer.add_summary(summary, self.training_step)
        return (accuracy, loss)

    def predict(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                                 self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=500, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="nli-dataset/nli-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="nli-dataset/nli-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="nli-dataset/nli-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--word_embedding", default=100, type=int, help="word_embedding")
    parser.add_argument("--char_embedding", default=100, type=int, help="char_embedding")
    parser.add_argument("--keep_prob", default=0.5, type=float, help="dropout probability")
    parser.add_argument("--num_filters", default=512, type=int, help="number of output filters from convolution")
    parser.add_argument("--l2", default=0.001, type=float, help="l2 lambda")

    args = parser.parse_args()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = nli_dataset.NLIDataset(args.data_train)
    data_dev = nli_dataset.NLIDataset(args.data_dev, train=data_train)
    data_test = nli_dataset.NLIDataset(args.data_test, train=data_train, no_languages=True)

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "{}-{}{}-bs{}-epochs{}-char{}-word{}-nf{}-l2:{}".format(tools.exp_name(__file__), args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs, args.char_embedding, args.word_embedding, args.num_filters, args.l2)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim,
                      num_words=len(data_train.vocabulary('words')), num_chars=len(data_train.vocabulary('chars')),
                      logdir=args.logdir, expname=expname, threads=args.threads,
                      word_embedding=args.word_embedding, char_embedding=args.char_embedding,
                      keep_prob=args.keep_prob, num_filters=args.num_filters, l2=args.l2)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_train.next_batch(args.batch_size)
            network.train(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, args.keep_prob)

        sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
            data_dev.whole_data_as_batch()
        dev_accuracy, dev_loss = network.evaluate(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, "dev")
        print("Development accuracy after epoch {} is {:.2f}. Dev loss is {:.2f}".format(epoch + 1, 100. * dev_accuracy, dev_loss), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_test.whole_data_as_batch()
            test_predictions = network.predict(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens)

    # Print test predictions
    for prediction in test_predictions:
        print(data_test.vocabulary('languages')[prediction])
