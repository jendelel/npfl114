from __future__ import division
from __future__ import print_function

import csv
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

class Network:
    OBSERVATIONS = 4
    LABELS = [0, 1]

    def __init__(self, threads=1, logdir=None, expname=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.summary_writer = tf.train.SummaryWriter(("{}/{}-{}" if expname else "{}/{}").format(logdir, timestamp, expname), flush_secs=10)
        else:
            self.summary_writer = None

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, observations, labels, summaries=True, run_metadata=True, keep_prob=0.5):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {self.observations: observations, self.labels: labels, self.keep_prob: keep_prob}}
        targets = [self.training]
        if summaries:
            targets.append(self.summaries["training"])
        if run_metadata:
            args["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            args["run_metadata"] = tf.RunMetadata()

        results = self.session.run(targets, **args)
        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step - 1)
        if run_metadata:
            self.summary_writer.add_run_metadata(args["run_metadata"], "step{:05}".format(self.training_step - 1))

    def construct(self, hidden_layer_size=800):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
                self.labels = tf.placeholder_with_default([], [None], name="labels")
            self.keep_prob = tf.placeholder_with_default(1.0, [], name="keep_prob")
            self.intlabels = tf.cast(self.labels, dtype=tf.int64)

            hidden_layer = tf_layers.fully_connected(self.observations, num_outputs=hidden_layer_size,
                                                     activation_fn=tf.nn.relu, scope="hidden_layer")
            hidden_layer_drop = tf.nn.dropout(hidden_layer, keep_prob=self.keep_prob)
            output_layer = tf_layers.fully_connected(hidden_layer_drop, num_outputs=len(self.LABELS), activation_fn=None,
                                                     scope="output_layer")
            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.intlabels, scope="loss")
            self.action = tf.argmax(output_layer, 1)
            self.accuracy = tf_metrics.accuracy(self.action, self.intlabels)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss", loss),
                                                            tf.scalar_summary("train/accuracy", self.accuracy)])}

            # Global step
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/action", self.action)
            self.saver = tf.train.Saver(max_to_keep=None)

            # Initialize the variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    # Save the graph
    def save(self, path):
        self.saver.save(self.session, path)


def load_cartpole_data(filename):
    features = []
    labels = []
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            features.append(row[:-1])
            labels.append(int(row[-1]))
    return features, labels

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-gym-save", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--epochs", default=300, type=int, help="Epoch number.")
    args = parser.parse_args()

    # Get the data
    observations, labels = load_cartpole_data(filename="labs04/gym-cartpole-data.txt")

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    network.construct(hidden_layer_size=200)

    # Train
    for _ in range(args.epochs):
        network.train(observations, labels, keep_prob=0.5)
    # Save the network
    network.save(args.exp)

