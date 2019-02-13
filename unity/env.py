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


class UnityAgentEnvironment(AgentEnvironment):
    def __init__(self, env_loc, train_mode=False):
        """Create new Unity environment.

        Params
        ======
            env_loc (string): location of the Unity environment.
            train_mode (boolean): determines if we're training.
        """
        self.env = UnityEnvironment(file_name=env_loc)
        self.train_mode = train_mode
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = len(state)
    def info(self):
        return self.state_size, self.action_size
    def step(self, action):
        assert 0 <= action <= self.action_size
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done
    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        return state
    def close(self):
        self.env.close()