'''
FilePath: /MAgIC-RL/magic_rl/agents/ppo_agent.py
Date: 2022-12-18 11:27:46
LastEditTime: 2022-12-18 11:45:37
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

import gymnasium as gym

from magic_rl.buffers.buffer import ReplayBuffer
from magic_rl.networks.network import CriticNetwork, ActorNetwork
from magic_rl.utils.gym_utils import NormalizeActions


class PpoAgent(object):
    '''

    '''

    def __init__(self, state_dim, action_dim, hidden_dim=128, method='clip'):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, ACTION_RANGE).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)
        print(self.actor, self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=A_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=C_LR)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []

    def get_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        s = s[np.newaxis, :].astype(np.float32)
        s = torch.Tensor(s).to(device)
        mean, std = self.actor(s)
        if greedy:
            a = mean.cpu().detach().numpy()[0]
        else:
            pi = torch.distributions.Normal(mean, std)
            a = pi.sample().cpu().numpy()[0]
        return np.clip(a, -self.actor.action_range, self.actor.action_range)

    def a_train(self, state, action, adv, old_pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        mu, sigma = self.actor(state)
        pi = torch.distributions.Normal(mu, sigma)
        ratio = torch.exp(pi.log_prob(action) - old_pi.log_prob(action))
        surr = ratio * adv
        if self.method == 'penalty':
            kl = torch.distributions.kl_divergence(old_pi, pi)
            kl_mean = kl.mean()
            aloss = -(surr - self.lam * kl).mean()
        else:  # clipping method, find this is better
            aloss = -torch.mean(
                torch.min(
                    surr,
                    torch.clamp(
                        ratio,
                        1. - self.epsilon,
                        1. + self.epsilon
                    ) * adv
                )
            )
        self.actor_opt.zero_grad()
        aloss.backward()
        self.actor_opt.step()

        if self.method == 'kl_pen':
            return kl_mean

    def c_train(self, cumulative_r, state):
        """
        Update actor network
        :param cumulative_r: cumulative reward batch
        :param state: state batch
        :return: None
        """
        advantage = cumulative_r - self.critic(state)
        closs = (advantage ** 2).mean()
        self.critic_opt.zero_grad()
        closs.backward()
        self.critic_opt.step()

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        s = torch.Tensor(self.state_buffer).to(device)
        a = torch.Tensor(self.action_buffer).to(device)
        r = torch.Tensor(self.cumulative_reward_buffer).to(device)
        with torch.no_grad():
            mean, std = self.actor(s)
            pi = torch.distributions.Normal(mean, std)
            adv = r - self.critic(s)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        if self.method == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv, pi)
                if kl > 4 * self.kl_target:  # this in in google's paper
                    break
            if kl < self.kl_target / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
            self.lam = np.clip(
                self.lam, 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv, pi)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(torch.Tensor([next_state]).to(device)).cpu().detach().numpy()[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_   # no future reward if next state is terminal
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

    def save_model(self):
        pass

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

