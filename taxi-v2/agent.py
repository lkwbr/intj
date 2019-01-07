from collections import defaultdict

import numpy as np


class Agent:
    def __init__(self, nA=6):
        """Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.i_episode = 1              # Episode number
        self.gamma = 0.9                # Future discount rate
        self.alpha = 0.2                # Learning rate
        self.alpha_decay_rate = 0.00001

    def _generate_policy(self, state):
        """Generate epsilon-greedy policy."""
        epsilon = 1 / self.i_episode ** 1.0
        policy = np.ones(self.nA) * epsilon / self.nA
        greedy_action = np.argmax(self.Q[state])
        policy[greedy_action] = 1 - epsilon + epsilon / self.nA
        return policy

    def select_action(self, state, epsilon_greedy=False):
        """Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if epsilon_greedy:
            # Epsilon-greedy policy
            policy = self._generate_policy(state)
            return np.random.choice(self.nA, p=policy)
        else:
            # Greedy policy.
            return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        """Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Update Q-table using Expected Sarsa.
        expected_sarsa_reward = np.dot(self.Q[next_state], self._generate_policy(next_state))
        sarsa_max_reward = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma * sarsa_max_reward - self.Q[state][action])
        if done:
            self.alpha = max(0, self.alpha - self.alpha_decay_rate)
            self.i_episode += 1