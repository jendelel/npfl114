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

    def __init__(self, rnn_cell, rnn_cell_dim, num_words, num_chars, logdir, expname, threads=1, seed=42, word_embedding=100, char_embedding=100, keep_prob=0.5, num_filters=512):
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

            input_chars = self.charseqs
            input_chars = tf.nn.embedding_lookup(tf.get_variable("char_emb", shape=[num_chars, char_embedding]), input_chars)
            input_chars = tf.expand_dims(input_chars, axis=3)
            print("input_chars", input_chars.get_shape())

            num_filters_c = int(char_embedding/4)
            cc3 = self._1d_conv(input_chars, num_outputs=num_filters_c, kernel_size=[3, char_embedding], stride=1)
            print("cc3", cc3.get_shape())
            cc4 = self._1d_conv(input_chars, num_outputs=num_filters_c, kernel_size=[4, char_embedding], stride=1)
            print("cc4", cc4.get_shape())
            cc5 = self._1d_conv(input_chars, num_outputs=num_filters_c, kernel_size=[5, char_embedding], stride=1)
            print("cc5", cc5.get_shape())
            cc7 = self._1d_conv(input_chars, num_outputs=num_filters_c, kernel_size=[7, char_embedding], stride=1)
            print("cc7", cc7.get_shape())

            pooled = []
            cp = tf.reduce_max(cc3, axis=1)
            print("c_pool", cp.get_shape())
            pooled.append(cp)
            pooled.append(tf.reduce_max(cc4, axis=1))
            pooled.append(tf.reduce_max(cc5, axis=1))
            pooled.append(tf.reduce_max(cc7, axis=1))

            c_pooled_outputs = tf.squeeze(tf.concat(2, pooled), axis=1)
            print("c_pooled_outputs", c_pooled_outputs.get_shape())

            w_slice = 20
            input_char_words = tf.nn.embedding_lookup(c_pooled_outputs, self.charseq_ids)
            input_char_words = tf.slice(input_char_words, [0, 0, 0], [-1, w_slice, char_embedding])
            print("input_char_words", input_char_words.get_shape())

            input_words = tf.nn.embedding_lookup(tf.get_variable("word_emb", shape=[num_words, word_embedding]), self.word_ids)
            input_words = tf.slice(input_words, [0, 0, 0], [-1, w_slice, word_embedding])
            print("input_words", input_words.get_shape())
            inputs = tf.pack([input_char_words, input_words], axis=3)
            print("inputs", inputs.get_shape())

            essays = tf.nn.embedding_lookup(inputs, self.sentence_ids)

            print("essays", essays.get_shape())

            seq_len = 33
            sliced = tf.slice(essays, [0, 0, 0, 0, 0], [-1, seq_len, w_slice, char_embedding, 2])
            print("sliced", sliced.get_shape())

            seq_len *= w_slice
            sliced = tf.reshape(sliced, [-1, seq_len, char_embedding, 2]) # TODO: convolution per sentence
            print("sliced", sliced.get_shape())

            c3 = self._1d_conv(sliced, num_outputs=num_filters, kernel_size=[3, char_embedding], stride=1)
            print("c3", c3.get_shape())
            c4 = self._1d_conv(sliced, num_outputs=num_filters, kernel_size=[4,char_embedding], stride=1)
            print("c4", c4.get_shape())
            c5 = self._1d_conv(sliced, num_outputs=num_filters, kernel_size=[5,char_embedding], stride=1)
            print("c5", c5.get_shape())
            c7 = self._1d_conv(sliced, num_outputs=num_filters, kernel_size=[7,char_embedding], stride=1)
            print("c7", c7.get_shape())

            pooled = []
            mp = tf_layers.max_pool2d(inputs=c3, kernel_size=[seq_len - 3 + 1, 1], stride=1)
            print("pool", mp.get_shape())
            pooled.append(mp)
            pooled.append(tf_layers.max_pool2d(inputs=c4, kernel_size=[seq_len - 4 + 1, 1], stride=1))
            pooled.append(tf_layers.max_pool2d(inputs=c5, kernel_size=[seq_len - 5 + 1, 1], stride=1))
            pooled.append(tf_layers.max_pool2d(inputs=c7, kernel_size=[seq_len - 7 + 1, 1], stride=1))


            pooled_outputs = tf.concat(3, pooled)
            print("pooled_outputs", pooled_outputs.get_shape())
            flatten = tf.reshape(pooled_outputs, [-1, 4 * num_filters])
            print("flatten", flatten.get_shape())
            dropout = tf_layers.dropout(flatten, keep_prob=keep_prob, is_training=self.is_training)

            # dec_outputs, _ = tf.nn.seq2seq.attention_decoder([dec_inputs], state_fw + state_bw, outputs, rnn_cell_co,
            #                                                  loop_function=loop_fn, output_size=self.LANGUAGES)
            output_layer = tf_layers.fully_connected(dropout, num_outputs=self.LANGUAGES)
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
                    {self.essay_lens: essay_lens, self.sentence_ids: sentence_ids, self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: "train", self.is_training: True})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, languages, dataset):
        accuracy, loss, summary = \
            self.session.run([self.accuracy, self.loss, self.summary],
                    {self.essay_lens: essay_lens, self.sentence_ids: sentence_ids, self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                              self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.languages: languages, self.dataset_name: dataset})
        self.summary_writer.add_summary(summary, self.training_step)
        return (accuracy, loss)

    def predict(self, essay_lens, sentence_ids, sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens):
        return self.session.run(self.predictions,
                {self.essay_lens: essay_lens, self.sentence_ids: sentence_ids, self.sentence_lens: sentence_lens, self.word_ids: word_ids,
                                 self.charseq_ids: charseq_ids, self.charseqs: charseqs, self.charseq_lens: charseq_lens})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=200, type=int, help="Batch size.")
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

    args = parser.parse_args()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = nli_dataset.NLIDataset(args.data_train)
    data_dev = nli_dataset.NLIDataset(args.data_dev, train=data_train)
    data_test = nli_dataset.NLIDataset(args.data_test, train=data_train, no_languages=True)

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "{}-{}{}-bs{}-epochs{}-char{}-word{}".format(tools.exp_name(__file__), args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs, args.char_embedding, args.word_embedding)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim,
                      num_words=len(data_train.vocabulary('words')), num_chars=len(data_train.vocabulary('chars')),
                      logdir=args.logdir, expname=expname, threads=args.threads,
                      word_embedding=args.word_embedding, char_embedding=args.char_embedding,
                      keep_prob=args.keep_prob, num_filters=args.num_filters)

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
