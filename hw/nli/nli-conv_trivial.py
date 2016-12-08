#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

import nli_dataset


class Network:
    LANGUAGES = 11

    def __init__(self, rnn_cell, rnn_cell_dim, num_words, num_chars, logdir, expname, batch_size, threads=1, seed=42, word_embedding=-1, keep_prob=0.5):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
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

            if word_embedding == -1:
                input_words = tf.one_hot(self.word_ids, num_words)
            else:
                input_words = tf.nn.embedding_lookup(tf.get_variable("word_emb", shape=[num_words, word_embedding]),
                                                     self.word_ids)
            print("input words", input_words.get_shape())

            (outputs_fw, outputs_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, input_words,
                                                                          self.sentence_lens, dtype=tf.float32)
            # print("state_bw", state_bw.get_shape())
            # print("state_fw", state_fw.get_shape())
            # state = state_bw + state_fw

            print("outputs_bw", outputs_bw.get_shape())
            print("outputs_fw", outputs_fw.get_shape())

            outputs = tf.pack([outputs_fw, outputs_bw], axis=3)
            print("outputs", outputs.get_shape())

            # essay_max_len(~320) x rnn_dim x 2
            conv_1 = tf_layers.convolution2d(inputs=outputs, num_outputs=32, kernel_size=[3, rnn_cell_dim], stride=1,
                                             activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm)
            conv_2 = tf_layers.convolution2d(inputs=conv_1, num_outputs=32, kernel_size=[3, rnn_cell_dim], stride=1,
                                             activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm)
            mp_1 = tf_layers.max_pool2d(inputs=conv_2, kernel_size=[4, 2], stride=2)

            print("mp_1", mp_1.get_shape())

            # essay_max_len/2 (~80) x rnn_dim/2 x 32
            conv_3 = tf_layers.convolution2d(inputs=mp_1, num_outputs=64, kernel_size=[3, rnn_cell_dim/2], stride=1,
                                             activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm)
            conv_4 = tf_layers.convolution2d(inputs=conv_3, num_outputs=64, kernel_size=[3, rnn_cell_dim/2], stride=1,
                                             activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm)
            mp_2 = tf_layers.max_pool2d(inputs=conv_4, kernel_size=[4, 4], stride=2)

            print("mp_2", mp_2.get_shape())

            # essay_max_len/4 (~20) x rnn_dim/4 (~25) x 64
            conv_5 = tf_layers.convolution2d(inputs=mp_2, num_outputs=128, kernel_size=[3, rnn_cell_dim/4], stride=1,
                                             activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm)
            conv_6 = tf_layers.convolution2d(inputs=conv_5, num_outputs=128, kernel_size=[3, rnn_cell_dim/4], stride=1,
                                             activation_fn=tf.nn.relu, normalizer_fn=tf_layers.batch_norm)
            mp_3 = tf_layers.max_pool2d(inputs=conv_6, kernel_size=[2, 5], stride=2)
            print("mp_3", mp_3.get_shape())

            (outputs2_fw, outputs2_bw), (state2_fw, state2_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell,
                                                                                             mp_3, dtype=tf.float32)

            packed_state = tf.pack([state2_fw, state2_bw], axis=2)
            print("packed_state", packed_state.get_shape())
            # flatten = tf_layers.flatten()
            # print("flatten", flatten.get_shape())
            fc_drop_1 = tf_layers.dropout(packed_state, keep_prob=keep_prob, is_training=self.is_training)
            fc = tf_layers.fully_connected(inputs=fc_drop_1, num_outputs=1024, activation_fn=tf.nn.relu)
            fc_drop_2 = tf_layers.dropout(fc, keep_prob=keep_prob, is_training=self.is_training)

            output_layer = tf_layers.fully_connected(fc_drop_2, num_outputs=self.LANGUAGES)
            print("output_layer", output_layer.get_shape())

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.languages)
            self.training = tf.train.AdamOptimizer().minimize(loss, self.global_step)
            self.predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.languages)

            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.merge_summary([tf.scalar_summary(self.dataset_name+"/loss", loss),
                                             tf.scalar_summary(self.dataset_name+"/accuracy", self.accuracy)])

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages):
        _, summary = \
            self.session.run([self.training, self.summary],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: "train", self.is_training: True})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, dataset):
        accuracy, summary = \
            self.session.run([self.accuracy, self.summary],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: dataset})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

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
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="nli-dataset/nli-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="nli-dataset/nli-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="nli-dataset/nli-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = nli_dataset.NLIDataset(args.data_train)
    data_dev = nli_dataset.NLIDataset(args.data_dev, train=data_train)
    data_test = nli_dataset.NLIDataset(args.data_test, train=data_train, no_languages=True)

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "nli-{}{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim,
                      num_words=len(data_train.vocabulary('words')), num_chars=len(data_train.vocabulary('chars')),
                      logdir=args.logdir, expname=expname, threads=args.threads, batch_size=args.batch_size)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_train.next_batch(args.batch_size)
            network.train(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages)

        sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
            data_dev.whole_data_as_batch()
        dev_accuracy = network.evaluate(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, "dev")
        print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                data_test.whole_data_as_batch()
            test_predictions = network.predict(sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens)

    # Print test predictions
    for prediction in test_predictions:
        print(data_test.vocabulary('languages')[prediction])
