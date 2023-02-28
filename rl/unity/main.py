#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from collections import namedtuple, deque
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import DqnAgent
from env import UnityAgentEnvironment


# TODO: Resolve issue with not being able to open environments in the
# same Python session.

# TODO: Use `statsmodel` and .yaml files to set network parameters. Makes it
# more modular.


def dqn_train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995, seed=7, plot=True):
    """Train a Deep Q-Network to play the Unity 'banana' environment."""
    print('TRAINING DQN')
    # Setup environment and agent.
    print('Loading environment...')
    env = UnityAgentEnvironment('.\Banana_Windows_x86_64\Banana.exe', True)
    state_size, action_size = env.info()
    print('Loading agent...')
    agent = DqnAgent(state_size, action_size, seed)
    # Run episodes.
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for _ in range(max_t):
            # Send action signal to environment and observe result.
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # Learn from that experience (SARSA tuple).
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        # Decay `epsilon` over time to hinder exploration.
        eps = max(eps_end, eps_decay*eps)
        # Tracking
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format( \
                                    i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format( \
                                            i_episode, np.mean(scores_window)))
    # Fin: print score and save agent policy.
    print("\rScore: {}".format(score))
    agent.save_policy()
    if plot:
        plot_scores('Training Scores', scores)
    env.close()
    return agent

def dqn_test(n_episodes=2, max_t=1000, seed=7, plot=True):
    """Test a DQN in the Unity 'banana' environment."""
    print('TESTING DQN')
    # Setup environment and agent.
    print('Loading environment...')
    env = UnityAgentEnvironment('.\Banana_Windows_x86_64\Banana.exe', False)
    state_size, action_size = env.info()
    print('Loading agent...')
    agent = DqnAgent(state_size, action_size, seed, True)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        scores.append(0)
        for _ in range(max_t):
            action = agent.act(state)
            state, reward, done = env.step(action)
            if done:
                break
            scores[-1] += reward
        print('Episode %d score: %f' % (i_episode, scores[-1]))
    print('Average score: %f' % np.mean(np.array(scores)))
    if plot:
        plot_scores('Testing Scores', scores)
    env.close()

def plot_scores(title, scores):
    plt.title(title)
    plt.title('DQN', loc='left')
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    """Entry point into program."""
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train and test a DQN agent.')
    parser.add_argument('--train', help='Train the agent.', \
                        action='store_true')
    parser.add_argument('--test', help='Test the agent.', \
                        action='store_true')
    args = parser.parse_args()
    if args.train:
        _ = dqn_train()
    if args.test:
        dqn_test()


if __name__ == '__main__':
    main()