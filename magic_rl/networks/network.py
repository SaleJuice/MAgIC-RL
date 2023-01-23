'''
FilePath: /MAgIC-RL/magic_rl/networks/network.py
Date: 2022-08-31 15:49:33
LastEditTime: 2023-01-23 19:13:58
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import os

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


def reset_parameters(linear:nn.Linear):
    stdv = 1. / math.sqrt(linear.weight.size(1))
    linear.weight.data.uniform_(-stdv, stdv)
    if linear.bias is not None:
        linear.bias.data.uniform_(-stdv, stdv)


# network models used by SAC-v2 algorithm:
# ----------------------------------------

class CriticNetwork(nn.Module):
    '''
    Attention: this is critic network for continuous observation continuous action(COCA).
    '''
    def __init__(self, state_dims:int, action_dims:int, hidden_size, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        
        # TODO to achieve more flexible hidden size
        # build architecture of network
        self.linear1 = nn.Linear(state_dims + action_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # TODO to achieve more autonomy random init w&b
        # init weight and bias of network
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of batch, which means state.shape[0] == action.shape[0].
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x  # value of this action under this state


class ActorNetwork(nn.Module):
    '''
    Attention: this is policy network for continuous observation continuous action(COCA).
    '''
    def __init__(self, state_dims:int, action_dims:int, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        
        # TODO to achieve more flexible hidden size
        # build architecture of network
        self.linear1 = nn.Linear(state_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_mean = nn.Linear(hidden_size, action_dims)
        self.linear_log_std = nn.Linear(hidden_size, action_dims)
        
        # TODO to achieve more autonomy random init w&b
        # init weight and bias of network
        self.linear_mean.weight.data.uniform_(-init_w, init_w)
        self.linear_mean.bias.data.uniform_(-init_w, init_w)

        self.linear_log_std.weight.data.uniform_(-init_w, init_w)
        self.linear_log_std.bias.data.uniform_(-init_w, init_w)

        # varibales
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.action_range = action_range
        self.num_actions = action_dims

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = F.leaky_relu(self.linear_mean(x))
        log_std = F.leaky_relu(self.linear_log_std(x))
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std  # mean and log(std) of the action under this state
        
# ----------------------------------------


if __name__ == '__main__':
    pass
