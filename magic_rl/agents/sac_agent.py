'''
FilePath: /MAgIC-RL/magic_rl/agents/sac_agent.py
Date: 2022-09-06 20:02:18
LastEditTime: 2022-09-23 16:55:42
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

from magic_rl.buffers.buffer import ReplayBuffer
from magic_rl.networks.network import CriticNetwork, ActorNetwork
from magic_rl.utils.gym_utils import NormalizeActions


class SacAgent(object):
    '''
    Soft Actor-Critic version 2
    using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
    add alpha loss compared with version 1
    paper: https://arxiv.org/pdf/1812.05905.pdf
    '''
    
    def __init__(self, state_dim, action_dim, hidden_dim, action_range, device):
        self.device = device

        self.soft_q_net1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_soft_q_net1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_soft_q_net2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = ActorNetwork(state_dim, action_dim, hidden_dim, action_range).to(self.device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=3e-4)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.policy_net(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(self.device)) # TanhNormal distribution as actions; reparameterization trick
        action = 1. * action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(1.)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean + std*z)
        action = torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        
        return action
    
    def update(self, batch_buff, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=0.005):  # FIXME soft_tau=1e-2
    # Sampling from replay buffer
        state, action, reward, next_state, done = batch_buff

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, _, _, _           = self.evaluate(state)
    
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        new_next_action, next_log_prob, _, _, _ = self.evaluate(next_state)
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = nn.MSELoss()(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = nn.MSELoss()(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        
        return q_value_loss1, q_value_loss2, policy_loss, alpha_loss

    def save_model(self, wb_dir):
        if not os.path.exists(wb_dir): os.makedirs(wb_dir)
        torch.save(self.soft_q_net1.state_dict(), os.path.join(wb_dir, "q1.pth"))
        torch.save(self.soft_q_net2.state_dict(), os.path.join(wb_dir, "q2.pth"))
        torch.save(self.policy_net.state_dict(), os.path.join(wb_dir, "policy.pth"))

    def load_model(self, wb_dir):
        assert os.path.exists(wb_dir), f"Directory '{wb_dir}' of weights and biases is not exist."
        self.soft_q_net1.load_state_dict(torch.load(os.path.join(wb_dir, "q1.pth")))
        self.soft_q_net2.load_state_dict(torch.load(os.path.join(wb_dir, "q2.pth")))
        self.policy_net.load_state_dict(torch.load(os.path.join(wb_dir, "policy.pth")))

    def set_train(self):
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.policy_net.train()

    def set_eval(self):
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


if __name__ == '__main__':
    # choose env
    # env = NormalizeActions(gym.make("Pendulum-v0"))
    # env = NormalizeActions(gym.make("LunarLanderContinuous-v2"))
    env = NormalizeActions(gym.make("Pendulum-v0"))

    # replay buffer
    replay_buffer = ReplayBuffer(1e6)

    # agent
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = 1.
    sac_agent = SacAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, replay_buffer=replay_buffer, action_range=action_range, device=torch.device("cuda:0"))

    # logger
    from torch.utils.tensorboard import SummaryWriter
    # sac_logger = SummaryWriter(log_dir="outputs/sac_v2/log/Pendulum-v1/3/")
    # sac_logger = SummaryWriter(log_dir="outputs/sac_v2/log/LunarLanderContinuous-v2/")
    sac_logger = SummaryWriter(log_dir="outputs/sac_v2/log/FishGym-v2/5/")

    # hyper-parameters for RL training
    batch_size  = 256

    all_steps = 0
    # training loop
    for episode in range(1_000_000+1):
        obs =  env.reset()
        print("env reset successed.")
        episode_reward = 0
        episode_len = 0
        for step in range(150):
            action = sac_agent.get_action(obs, deterministic = False)
            next_obs, reward, done, _ = env.step(action)
            # env.render()
                
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_len += 1
            all_steps += 1
            
            if len(replay_buffer) > batch_size:
                q_value_loss1, q_value_loss2, policy_loss, alpha_loss = sac_agent.update(batch_size, reward_scale=20., auto_entropy=True, target_entropy=-4)
                sac_logger.add_scalar("loss/q_value_loss1", q_value_loss1, all_steps)
                sac_logger.add_scalar("loss/q_value_loss2", q_value_loss2, all_steps)
                sac_logger.add_scalar("loss/policy_loss", policy_loss, all_steps)
                sac_logger.add_scalar("loss/alpha_loss", alpha_loss, all_steps)
                    
            if done:
                break
        
        sac_logger.add_scalar("episode/reward", episode_reward, all_steps)
        sac_logger.add_scalar("episode/len", episode_len, all_steps)
        print(f"Episode: {episode} | Episode Reward: {episode_reward}")
        # env.save_traj(suffix=f"{all_steps}".zfill(10))
        print("save img successed, waiting to reset env.")

