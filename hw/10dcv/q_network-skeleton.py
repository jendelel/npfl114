#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

from random import random, randint, seed

import environment_continuous
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

class QNetwork:
    def __init__(self, observations, actions, q_network, learning_rate, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        # Construct the graph
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, observations])

            self.q = q_network(self.observations)
            print("q", self.q.get_shape())
            self.action = tf.argmax(self.q, 1)
            print("action", self.action.get_shape())

            self.target_q = tf.placeholder(tf.float32, [None, actions])
            print("target_q", self.target_q.get_shape())

            loss = tf.reduce_mean(tf.square(tf.sub(self.target_q, self.q)))
            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    def predict(self, observations):
        return self.session.run([self.action, self.q],
                                {self.observations: observations})

    def train(self, observations, target_q):
        self.session.run(self.training,
                         {self.observations: observations,
                          self.target_q: target_q})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)
    seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Taxi-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=1500, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--alpha", default=0.02, type=float, help="Learning rate.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Epsilon.")
    parser.add_argument("--epsilon_final", default=0.001, type=float, help="Epsilon decay rate.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = environment_continuous.EnvironmentContinuous(args.env)
    if args.render_each:
        # Because of low TLS limit, load OpenGL before TensorFlow
        env.reset()
        env.render()

    # Create policy network
    def q_network(observations):
        return tf_layers.linear(observations, env.actions, biases_initializer=None)
    qn = QNetwork(observations=env.observations, actions=env.actions, q_network=q_network,
                  learning_rate=args.alpha, threads=args.threads)

    epsilon = args.epsilon
    episode_rewards, episode_lengths = [], []
    for episode in range(args.episodes):
        # Perform episode
        observation = env.reset()
        total_reward = 0
        for t in range(args.max_steps):
            if args.render_each and episode > 0 and episode % args.render_each == 0:
                env.render()

            [action], [q_values] = qn.predict(observations=[observation])
            q_values = q_values.tolist()

            if random() < epsilon:
                action = randint(0, env.actions-1)

            next_observation, reward, done, _ = env.step(action)
            total_reward += reward

            _, [next_q_values] = qn.predict(observations=[next_observation])
            next_q_values = next_q_values.tolist()

            target_q_values = list(q_values)
            target_q_values[action] = reward + args.gamma * max(next_q_values)

            # Train the QNetwork using qn.train
            qn.train([observation], [target_q_values])

            observation = next_observation
            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(t)
        if len(episode_rewards) % 10 == 0:
            print("Episode {}, mean 100-episode reward {}, mean 100-episode length {}, epsilon {}.".format(
                episode + 1, np.mean(episode_rewards[-100:]), np.mean(episode_lengths[-100:]), epsilon))

        if args.epsilon_final:
            epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
