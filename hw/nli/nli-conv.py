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
            mp = tf_layers.max_pool2d(inputs=inp, kernel_size=kernel_size, stride=stride)
            print("mp", mp.get_shape())
            return mp

    def _1d_conv(self, inp, num_outputs, kernel_size=3, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm):
        with self.session.graph.as_default():
            conv_1 = tf_layers.convolution2d(inputs=inp, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride,
                                             activation_fn=activation_fn, normalizer_fn=normalizer_fn)

            conv_2 = tf_layers.convolution2d(inputs=conv_1, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride,
                                             activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            conv_2 = conv_2 + conv_1  # residual connection
            print("Conv:", conv_2.get_shape())
            return conv_2

    def __init__(self, rnn_cell, rnn_cell_dim, num_words, num_chars, logdir, expname, threads=1, seed=42, word_embedding=100, char_embedding=100, keep_prob=0.5, rnn_num=0, l2=0.001):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            with tf.variable_scope("all", regularizer=tf_layers.l2_regularizer(l2)):
                if rnn_cell == "LSTM":
                    rnn_cell_co = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
                    rnn_cell_co2 = tf.nn.rnn_cell.LSTMCell(word_embedding+char_embedding)
                elif rnn_cell == "GRU":
                    rnn_cell_co = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
                    rnn_cell_co2 = tf.nn.rnn_cell.GRUCell(word_embedding+char_embedding)
                elif rnn_cell == "GRUGRID":
                    rnn_cell_co = tf.contrib.rnn.Grid2GRUCell
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
                self.tags = tf.placeholder(tf.int32, [None, None])

                # charseqs je matice [batch x id charakteru]
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
                input_tags = tf.nn.embedding_lookup(tf.get_variable("tag_emb", shape=[num_words, word_embedding]),
                                                    self.tags)
                input_concat = tf.concat(2, [input_char_words, input_words, input_tags])
                seq_len = 250
                input_concat = tf.slice(input_concat, [0, 0, 0], [-1, seq_len, char_embedding+word_embedding+word_embedding])
                input_concat = tf.expand_dims(input_concat, -1)
                print("input_concat", input_concat.get_shape())

                num_filters = 128
                filters = [3, 4, 5]
                pooled = []
                for i, filter_size in enumerate(filters):
                    conv = tf_layers.convolution2d(input_concat, num_outputs=num_filters, kernel_size=filter_size, stride=1, activation_fn=tf.nn.relu,
                            normalizer_fn=tf_layers.batch_norm)
                    mp = self._max_pool(conv, kernel_size=seq_len, stride=1)
                    pooled.append(mp)

                num_filters_total = num_filters * len(filters)
                pool = tf.concat(3, pooled)
                print("pool", pool.get_shape())
                pool_r = tf.squeeze(pool, axis=1)
                print("pool_r", pool_r.get_shape())
                dim2 = int(pool_r.get_shape()[1])
                pool_flat = tf.reshape(pool_r, [-1, dim2*num_filters_total])
                print("pool_flat", pool_flat.get_shape())
                drop = tf_layers.dropout(pool_flat, keep_prob=keep_prob, is_training=self.is_training)
                output_layer = tf_layers.fully_connected(drop, num_outputs=self.LANGUAGES)
                print("output_layer", output_layer.get_shape())

                loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.languages)
                self.training = tf.train.AdamOptimizer().minimize(loss, self.global_step)
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

    def train(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, tags):
        _, summary = \
            self.session.run([self.training, self.summary],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: "train", self.is_training: True, self.tags: tags})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, tags, dataset):
        accuracy, summary = \
            self.session.run([self.accuracy, self.summary],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: dataset, self.tags: tags})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def predict(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                                 self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                                 self.tags: tags})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=25, type=int, help="Batch size.")
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
    parser.add_argument("--rnn_num", default=0, type=int, help="number of rnns")
    parser.add_argument("--keep_prob", default=0.5, type=float, help="dropout keep prob")
    parser.add_argument("--l2", default=0.001, type=float, help="l2 lambda")

    args = parser.parse_args()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = nli_dataset.NLIDataset(args.data_train)
    data_dev = nli_dataset.NLIDataset(args.data_dev, train=data_train)
    data_test = nli_dataset.NLIDataset(args.data_test, train=data_train, no_languages=True)

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "{}-{}{}-bs{}-epochs{}-char{}-word{}-rnn{}-drop{}-l2{}".format(tools.exp_name(__file__), args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs, args.char_embedding, args.word_embedding, args.rnn_num, args.keep_prob, args.l2)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim,
                      num_words=len(data_train.vocabulary('words')), num_chars=len(data_train.vocabulary('chars')),
                      logdir=args.logdir, expname=expname, threads=args.threads, keep_prob=args.keep_prob,
                      word_embedding=args.word_embedding, char_embedding=args.char_embedding, rnn_num=args.rnn_num, l2=args.l2)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_train.next_batch(args.batch_size)
            network.train(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, tags)

        sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
            data_dev.whole_data_as_batch()
        dev_accuracy = network.evaluate(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, tags, "dev")
        print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_test.whole_data_as_batch()
            test_predictions = network.predict(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags)

    # Print test predictions
    for prediction in test_predictions:
        print(data_test.vocabulary('languages')[prediction])
