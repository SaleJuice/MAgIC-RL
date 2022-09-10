'''
FilePath: /torch-rl/network.py
Date: 2022-08-31 15:49:33
LastEditTime: 2022-09-10 19:32:22
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


# network models used by SAC-v2 algorithm:
# ----------------------------------------

class CriticNetwork(nn.Module):
    '''
    Attention: this is critic network for continuous observation continuous action(COCA).
    '''
    def __init__(self, state_dims:int, action_dims:int, hidden_size, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        # build architecture of network
        # XXX to achieve more flexible hidden size
        self.linear1 = nn.Linear(state_dims + action_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        # init weight and bias of network
        # FIXME check the effect of these code
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of batch, which means state.shape[0] == action.shape[0].
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x  # value of this action under this state


class PolicyNetwork(nn.Module):
    '''
    Attention: this is policy network for continuous observation continuous action(COCA).
    '''
    def __init__(self, state_dims:int, action_dims:int, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        # build architecture of network
        # XXX to achieve more flexible hidden size
        self.linear1 = nn.Linear(state_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        
        # init weight and bias of network
        # FIXME to check the effect of these code
        self.linear_mean = nn.Linear(hidden_size, action_dims)
        self.linear_mean.weight.data.uniform_(-init_w, init_w)
        self.linear_mean.bias.data.uniform_(-init_w, init_w)

        self.linear_log_std = nn.Linear(hidden_size, action_dims)
        self.linear_log_std.weight.data.uniform_(-init_w, init_w)
        self.linear_log_std.bias.data.uniform_(-init_w, init_w)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.action_range = action_range
        self.num_actions = action_dims

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        # TODO to check the effect of using 'mean = F.leaky_relu(self.mean_linear(x))'
        mean = self.linear_mean(x)
        log_std = self.linear_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std  # mean and log(std) of the action under this state
        
# ----------------------------------------


if __name__ == '__main__':
    pass
