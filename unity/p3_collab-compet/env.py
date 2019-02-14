#!/usr/bin/python
# -*- coding: utf-8 -*-

import abc

from unityagents import UnityEnvironment


class AgentEnvironment(abc.ABC):
    """Abstract agent environment."""
    @abc.abstractmethod
    def info(self):
        """
        Get environment information.

        Returns
        ======
            state_size (int): maximum number of training episodes.
            action_size (int): maximum number of timesteps per episode.
        """
        pass
    @abc.abstractmethod
    def step(self, action):
        """
        Perform given `action` in environment.

        Params
        ======
            action (int): maximum number of training episodes

        Returns
        ======
            state (int): new state resulting from given `action`.
            reward (float): reward for being in the new `state`.
            done (boolean): indicates end of current episode.
        """
        pass
    @abc.abstractmethod
    def reset(self):
        """
        Reset the environement for the next episode.

        Returns
        ======
            state (int): starting state in refreshed environment.
        """
        pass
    @abc.abstractmethod
    def close(self):
        """Close environment."""
        pass


class ContinuousUnityAgentEnvironment(AgentEnvironment):
    def __init__(self, env_loc, train_mode=False, no_graphics=False):
        """Create new continuous, multi-agent Unity environment.

        Params
        ======
            env_loc (string): location of the Unity environment.
            train_mode (boolean): determines if we're training.
            no_graphics (boolean): decides if showing graphics.
        """
        self.env = UnityEnvironment(file_name=env_loc, no_graphics=no_graphics)
        self.train_mode = train_mode
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        self.state_size = len(env_info.vector_observations[0])
        self.action_size = self.brain.vector_action_space_size
        self.n_agents = len(env_info.agents)
    def info(self):
        return self.state_size, self.action_size, self.n_agents
    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return next_states, rewards, dones
    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        states = env_info.vector_observations
        return states
    def close(self):
        self.env.close()