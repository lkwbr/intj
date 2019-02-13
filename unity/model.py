#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    FC1_UNITS = 64
    FC2_UNITS = 128
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self._init_nn()
    def _init_nn(self):
        """Initialize fully-connected neural network."""
        self.fc1 = torch.nn.Linear(self.state_size, QNetwork.FC1_UNITS)
        self.fc2 = torch.nn.Linear(QNetwork.FC1_UNITS, QNetwork.FC2_UNITS)
        self.fc3 = torch.nn.Linear(QNetwork.FC2_UNITS, self.action_size)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x