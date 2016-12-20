#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import environment_discrete
import numpy as np
from random import random, randint, seed

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)
    seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=500, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.5, type=float, help="Epsilon.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Epsilon decay rate.")
    parser.add_argument("--gamma", default=0.99999, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = environment_discrete.EnvironmentDiscrete(args.env)

    # Create Q, C and other variables
    Q = np.zeros([env.states, env.actions])
    C = np.zeros([env.states, env.actions])
    steps = np.zeros([env.states, env.actions])
    epsilon = args.epsilon
    episode_rewards, episode_lengths = [], []

    for episode in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards, total_reward = [], [], [], 0
        t = 0
        for t in range(args.max_steps):
            if args.render_each and episode > 0 and episode % args.render_each == 0:
                env.render()

            if random() < epsilon:
                action = randint(0, env.actions-1)
            else:
                action = Q[state].argmax()

            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        for i in range(t):
            g_i = 0
            for j in range(t-i):
                g_i += rewards[i+j]*(args.gamma**(j-1))
            C[states[i], actions[i]] += g_i
            steps[states[i], actions[i]] += 1
        for s in states:
            for a in actions:
                num_steps = steps[s, a]
                if num_steps == 0:
                    Q[s, a] = 0
                else:
                    Q[s, a] = C[s, a]/num_steps
        episode_rewards.append(total_reward)
        episode_lengths.append(t)
        if len(episode_rewards) % 10 == 0:
            print("Episode {}, mean 100-episode reward {}, mean 100-episode length {}, epsilon {}.".format(
                episode + 1, np.mean(episode_rewards[-100:]), np.mean(episode_lengths[-100:]), epsilon))

        if args.epsilon_final:
            epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
