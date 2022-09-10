'''
FilePath: /MAgIC-RL/train.py
Date: 2022-09-10 22:01:43
LastEditTime: 2022-09-10 22:02:48
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import os
import time
import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import gym

from buffer import ReplayBuffer
from network import CriticNetwork, PolicyNetwork
from agent import SacAgent
from utils import NormalizedActions


if __name__ == '__main__':
    # choose env
    env = NormalizedActions(gym.make("Pendulum-v1"))
    # env = NormalizedActions(gym.make("LunarLanderContinuous-v2"))

    # replay buffer
    replay_buffer = ReplayBuffer(1e6)

    # agent
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = 1.
    sac_agent = SacAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=512, replay_buffer=replay_buffer, action_range=action_range, device=torch.device("cuda:0"))

    # logger
    from torch.utils.tensorboard import SummaryWriter
    sac_logger = SummaryWriter(log_dir="outputs/sac_v2/log/Pendulum-v1/")
    # sac_logger = SummaryWriter(log_dir="outputs/sac_v2/log/LunarLanderContinuous-v2/")

    # hyper-parameters for RL training
    batch_size  = 300

    # training loop
    for episode in range(1000+1):
        obs =  env.reset()
        episode_reward = 0
        for step in range(150):
            action = sac_agent.get_action(obs, deterministic = False)
            next_obs, reward, done, _ = env.step(action)
            # env.render()
                
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                q_value_loss1, q_value_loss2, policy_loss = sac_agent.update(batch_size, reward_scale=10., auto_entropy=False)
                sac_logger.add_scalar("q_value_loss1", q_value_loss1, episode)
                sac_logger.add_scalar("q_value_loss2", q_value_loss2, episode)
                sac_logger.add_scalar("policy_loss", policy_loss, episode)
                    
            if done:
                break
        
        sac_logger.add_scalar("episode reward", episode_reward, episode)
        print(f"Episode: {episode} | Episode Reward: {episode_reward}")
