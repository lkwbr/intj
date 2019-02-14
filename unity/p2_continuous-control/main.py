#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from collections import deque

from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import ActorCriticAgent
from env import ContinuousUnityAgentEnvironment


def ddpg_train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01,
               eps_decay=0.995, seed=7, plot=True):
    """Train DDPG agent."""
    # Step environment and agent.
    print('TRAINING DDPG')
    env = ContinuousUnityAgentEnvironment(
        './Reacher_Windows_x86_64/Reacher.exe', True)
    state_size, action_size, n_agents = env.info()
    agent = ActorCriticAgent(state_size, n_agents, action_size, 4)
    scores = []
    scores_window = deque(maxlen=100)
    n_episodes = 1000
    for i_episode in range(n_episodes):
        states = env.reset()
        agent.reset() # reset noise
        score = np.zeros(n_agents)
        while True:
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break
        agent.checkpoint()
        scores.append(np.mean(score))
        scores_window.append(np.mean(score))
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(i_episode, np.mean(score), np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format( \
                                            i_episode, np.mean(scores_window)))
        # Stopping condition
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            break
    plot_scores('Training Scores', scores)
    return agent

def ddpg_test():
    """Test a trained DDPG agent."""
    print('TESTING DDPG')
    env = ContinuousUnityAgentEnvironment(
        './Reacher_Windows_x86_64/Reacher.exe', False)
    state_size, action_size, n_agents = env.info()
    agent = ActorCriticAgent(state_size, n_agents, action_size, 4)
    for episode in range(3):
        states = env.reset()
        score = np.zeros(n_agents)
        while True:
            actions = agent.act(states, add_noise=False)
            next_states, rewards, dones = env.step(actions)
            score += rewards
            states = next_states
            if np.any(dones):
                break
        print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))

def plot_scores(title, scores):
    plt.title(title)
    plt.title('DDPG', loc='left')
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    """Entry point of program."""
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train and test a DQN agent.')
    parser.add_argument('--train', help='Train the agent.', \
                        action='store_true')
    parser.add_argument('--test', help='Test the agent.', \
                        action='store_true')
    args = parser.parse_args()
    if args.train:
        _ = ddpg_train()
    if args.test:
        ddpg_test()


if __name__ == '__main__':
    main()