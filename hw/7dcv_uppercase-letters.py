#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

class Dataset:
    def __init__(self, filename, alphabet = None):
        # Load the sentences
        sentences = []
        with open(filename, "rw") as file:
            for line in file:
                sentences.append(line.rstrip("\r\n"))

        # Compute sentence lengths
        self._sentence_lens = np.zeros([len(sentences)], np.int32)
        for i in range(len(sentences)):
            self._sentence_lens[i] = len(sentences[i])
        max_sentence_len = np.max(self._sentence_lens)

        # Create alphabet_map
        alphabet_map = {'<pad>': 0, '<unk>': 1}
        if alphabet is not None:
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index

        # Remap input characters using the alphabet_map
        self._sentences = np.zeros([len(sentences), max_sentence_len], np.int32)
        self._labels = np.zeros([len(sentences), max_sentence_len], np.int32)
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                char = sentences[i][j].lower()
                if char not in alphabet_map:
                    if alphabet is None:
                        alphabet_map[char] = len(alphabet_map)
                    else:
                        char = '<unk>'
                self._sentences[i, j] = alphabet_map[char]
                self._labels[i, j] = 0 if sentences[i][j].lower() == sentences[i][j] else 1

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.iteritems():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._sentences))

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def sentences(self):
        return self._sentences

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        batch_len = np.max(self._sentence_lens[batch_perm])
        return self._sentences[batch_perm, 0:batch_len], self._sentence_lens[batch_perm], self._labels[batch_perm, 0:batch_len]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentences))
            return True
        return False


class Network:
    def __init__(self, alphabet_size, rnn_cell, rnn_cell_dim, embedding_dim, logdir, expname, threads=1, seed=42):
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
            self.sentences = tf.placeholder(tf.int32, [None, None])
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.labels = tf.placeholder(tf.int64, [None, None])
            
            input_words=None
            if embedding_dim == -1:
                input_words = tf.one_hot(self.sentences, alphabet_size)
            else:
                input_words = tf.nn.embedding_lookup(tf.get_variable("alpha_emb", shape=[alphabet_size, embedding_dim]), self.sentences)
            print("input words", input_words.get_shape())
            (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, input_words, self.sentence_lens, dtype=tf.float32)
           
            print("outputs_bw", outputs_bw.get_shape())
            print("outputs_fw", outputs_fw.get_shape())
            outputs = outputs_fw + outputs_bw
            print("input_words", input_words.get_shape())
            print("outputs", outputs.get_shape())
            mask = tf.cast(tf.sequence_mask(self.sentence_lens), tf.float32)
            mask3d = tf.pack(np.repeat(mask, rnn_cell_dim).tolist(), axis=2)
            print("mask3d", mask3d.get_shape())
            masked = mask3d * outputs
            print("masked", masked.get_shape())
            masked_vec = tf.reshape(masked, [-1, rnn_cell_dim])
            print("masked_vec", masked_vec.get_shape())
            labels_vec = tf.reshape(self.labels, [-1])
            print("labels_vec", labels_vec.get_shape())

            output_layer = tf_layers.fully_connected(masked_vec, 2)

            print("output_layer", output_layer.get_shape())
            self.predictions = tf.cast(tf.argmax(output_layer, 1),tf.int64)
            print("predictions", self.predictions.get_shape())
            print("labels", self.labels.get_shape())
            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, labels_vec)
            #loss = tf.reduce_sum((tf.sigmoid(outputs_vec)-tf.cast(labels_vec, tf.float32))**2)
            print("scalar", loss.get_shape())
            self.training = tf.train.AdamOptimizer().minimize(loss, self.global_step)
            self.accuracy = tf.contrib.metrics.accuracy(self.predictions, labels_vec)
            # rnn bunka implementovana dynamicky
            # while bunka v TF (iba s jednym TF node pre RNN) ... cele v
            # (outputs, state) tf.nn.dynamic_rnn (cell, inputs, seq_lens=None,
            # init_state=None .. default zero_state z ^^cell, time_major=False)
            # inputs: tensor[batch, sequence, input_size # one-hot vector, input representation #]
            #
            # ((outputs_fw, outputs_bw), (states_fw, states_bw))tf.bidirectional_dynamic_rnn(cell_fw, cell_bw, ...)

            # Embeddings pro pismenka taky...
            # tf.nn.embedding_lookup(weights (embeddings)...(tf.get_variable("alpha_emb", shape=[alphabet_size, dim_of_embedding])), indices (?dimenze?))
            #
            # Orezani outputs
            # tf.sequence_mask(sequence_lengths, max_seq_length) to float (tf.cast) and multiply!

            # TODO
            # accuracy = ...
            # self.training = ...

            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.scalar_summary(self.dataset_name+"/accuracy", self.accuracy)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentences, sentence_lens, labels):
        _,acc, summary = self.session.run([self.training, self.accuracy, self.summary],
                                      {self.sentences: sentences, self.sentence_lens: sentence_lens,
                                       self.labels: labels, self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)
        return acc

    def evaluate(self, sentences, sentence_lens, labels, dataset_name):
        acc, summary = self.session.run([self.accuracy, self.summary], {self.sentences: sentences, self.sentence_lens: sentence_lens,
                                                  self.labels: labels, self.dataset_name: dataset_name})
        self.summary_writer.add_summary(summary, self.training_step)
        return acc


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="en-ud-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="en-ud-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="en-ud-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--embedding", default=150, type=int, help="Embedding dimension. If -1, one_hot is used.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    data_train = Dataset(args.data_train)
    data_dev = Dataset(args.data_dev, data_train.alphabet)
    data_test = Dataset(args.data_test, data_train.alphabet)

    # Construct the network
    expname = "uppercase-letters-{}{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs)
    network = Network(alphabet_size=len(data_train.alphabet), rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim, embedding_dim=args.embedding, logdir=args.logdir, expname=expname, threads=args.threads)

    # Train
    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch))
        while not data_train.epoch_finished():
            sentences, sentence_lens, labels = data_train.next_batch(args.batch_size)
            network.train(sentences, sentence_lens, labels)

        network.evaluate(data_dev.sentences, data_dev.sentence_lens, data_dev.labels, "dev")
        network.evaluate(data_test.sentences, data_test.sentence_lens, data_test.labels, "test")
