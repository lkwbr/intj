#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from collections import deque
import json
import os

from unityagents import UnityEnvironment
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import ActorCriticAgents
from env import ContinuousUnityAgentEnvironment


matplotlib.use('Agg')


def ddpg_train(n_episodes=4000, max_t=2000, seed=10):
    env = ContinuousUnityAgentEnvironment('./Tennis_Windows_x86_64/Tennis.exe', True)
    state_size, action_size, num_agents = env.info()
    agent = ActorCriticAgents(state_size=state_size, action_size=action_size,
                   num_agents=num_agents, random_seed=seed)
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        states = env.reset()
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.3f}'.format(i_episode, np.mean(scores_deque), np.mean(score)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), f'actor_{i_episode}.pth')
            torch.save(agent.critic_local.state_dict(), f'critic_{i_episode}.pth')
        if i_episode % 500 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.savefig('scores_%d.png' % i_episode)
            plt.show()
    plot_scores('Training Scores', scores)
    env.close()

def plot_scores(title, scores, save=False):
    plt.title(title)
    plt.title('DDPG', loc='left')
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if save:
        plt.savefig('scores.png')
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
        _ = ddpg_train(n_episodes=100)
    # TODO: Testing code.

if __name__ == '__main__':
    main()