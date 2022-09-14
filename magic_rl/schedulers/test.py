'''
FilePath: /MAgIC-RL/magic_rl/schedulers/test.py
Date: 2022-09-07 11:29:49
LastEditTime: 2022-09-13 12:30:18
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


# network models used by SAC-v2 algorithm:
# ----------------------------------------

class CriticNetwork(nn.Module):
    '''
    Attention: this is critic network for continuous observation continuous action(COCA).
    Attributes:
        self.optimizer
    Methods:
        self.save_chkpt()
        self.load_chkpt()
    '''
    def __init__(self, 
        name:str, 
        chkpt_dir:str,
        state_dims:int,
        action_dims:int,
        optimizer:optim.Optimizer,
        lr:float,
        device:torch.device,
        ):
        super(CriticNetwork, self).__init__()
        # network
        self.linear1 = nn.Linear(state_dims+action_dims, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 1)

        # FIXME check the effect of these code
        # self.linear4.weight.data.uniform_(-3e-3, 3e-3)
        # self.linear4.bias.data.uniform_(-3e-3, 3e-3)
        
        # optimizer
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.to(device)

        # checkpoint
        self.name = name
        self.chkpt_dir = chkpt_dir

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # the dim 0 is number of batch, which means state.shape[0] == action.shape[0].
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x  # value of this action under this state

    def save_chkpt(self):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        torch.save(self.state_dict(), os.path.join(self.chkpt_dir, self.name))

    def load_chkpt(self, chkpt_file:str=None):
        if chkpt_file is None:
            chkpt_file = os.path.join(self.chkpt_dir, self.name)
        if os.path.exists(chkpt_file):
            self.load_state_dict(torch.load(chkpt_file))
        else:
            assert (False), f"checkpoint file '{chkpt_file}' do not exist!"
        
# ----------------------------------------

class DqnAgent(object):
    def __init__(self) -> None:
        # env part
        self.env = gym.make("CartPole-v0")
        # model part
        self.optimizer = None  # FIXME: 
        self.learning_rate = 1e-3
        self.q_net = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n, 64, self.learning_rate)
        # dpn part
        self.batch_size = None  # TODO: 
        self.buffer_size = 1000
        self.replay_buffer = Buffer(self.buffer_size)
        self.gamma = 0.99
        # others part
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def rollout(self, episode_times=1):
        for _ in range(episode_times):
            obs = self.env.reset()
            while True:
                act = self.env.action_space.sample()
                obs_next, rew, done, info = self.env.step(act)
                self.replay_buffer.add(obs, act, rew, done)
                obs = obs_next

                if done:
                    self.replay_buffer.add(obs, None, None, None)
                    break

    def train(self):
        # get a batch of transitions from replay buffer
        obs, act, rew, done, obs_next, _ = self.replay_buffer.sample()
        print(obs, act, rew, done, obs_next)
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        rew = torch.FloatTensor(rew).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        obs_next = torch.FloatTensor(obs_next).to(self.device)
        print(obs, act, rew, done, obs_next)
        exit()

        # use TD algorithm to update q_net
        q = self.q_net(obs)
        print(f"q: {q}")
        exit()
        q = self.q_net(state).gather(1, actions)
        td_target = self.q_net(next_states)
        q_target_max = self.q_net(next_states).max(1)[0].unsqueeze(1).detach()
        targets = rewards + self.gamma * q_target_max * dones
        q_out = self.q_net(states)
        q_a = q_out.gather(1, actions)

        # Multiply Importance Sampling weights to loss        
        loss = F.smooth_l1_loss(q_a, targets)
        
        # Update Network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    critic_net = CriticNetwork(
        name="critic_net.pth", 
        chkpt_dir="chkpt/",
        state_dims=4,
        action_dims=1,
        optimizer=optim.Adam,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    print(critic_net)
    critic_net.save_chkpt()
    critic_net.load_chkpt()


from typing import List, Dict
import numpy as np

from magic_rl.buffers.episode import Episode


def grab_red_envelope(num_people, amount) -> list:
        a = [np.random.randint(0, amount) for i in range(num_people-1)]
        a.append(0)
        a.append(amount)
        a.sort()
        b = [a[i+1]-a[i] for i in range(num_people)]
        return b


class Buffer(object):
    '''
    Attention: real buffer size(step_nums) is always smaller than buffer_size.
    '''

    def __init__(self, buffer_size:int) -> None:
        self.buffer_size = buffer_size
        self.episodes: List[Episode] = []

    def __len__(self) -> int:
        return self.step_nums

    def add(self, obs, act, rew, done) -> None:
        if self.episode_nums == 0:
            self.episodes.append(Episode())

        if self.episodes[-1].is_complete:
            self.episodes.append(Episode())

        self.episodes[-1].add(obs, act, rew, done)

        while self.step_nums > self.buffer_size:
            self.episodes.pop(0)

    def sample(self, batch_size:int=1) -> Dict[str, np.ndarray]:
        assert (self.episode_nums > 0), f"there is no episode can be sampled."
        assert (self.step_nums > 0), f"there is no steps can be sampled."

        batch = {}
        batch["obs"] = []
        batch["act"] = []
        batch["rew"] = []
        batch["done"] = []
        batch["obs_next"] = []

        indexs = grab_red_envelope(self.step_nums - 1, batch_size)

        if len(self.episodes[-1]) == 0:
            index = np.random.randint(self.episode_nums-1)
        else:
            index = np.random.randint(self.episode_nums)
        obs, act, rew, done, obs_next, act_next = self.episodes[index].sample()
        return obs, act, rew, done, obs_next, act_next

    @property
    def episode_nums(self) -> int:
        return len(self.episodes)

    @property
    def step_nums(self) -> int:
        return sum([len(episode) for episode in self.episodes])


if __name__ == "__main__":
    # import gym

    # buf = Buffer(buffer_size=1000)

    # env = gym.make("CartPole-v1")
    # for _ in range(10):
    #     obs = env.reset()
    #     while True:
    #         act = env.action_space.sample()
    #         obs_next, rew, done, info = env.step(act)
    #         buf.add(obs, act, rew, done)
    #         obs = obs_next

    #         if done:
    #             buf.add(obs, None, None, None)
    #             break

    # print(f"buffer_size: {buf.buffer_size}")
    # print(f"step_nums: {buf.step_nums}")
    # print(f"episode_nums: {buf.episode_nums}")
    # print(f"each episode length: {[len(episode) for episode in buf.episodes]}")
    # print(f"each episode is_complete: {[episode.is_complete for episode in buf.episodes]}")
    # print(f"each episode is_samplable: {[episode.is_samplable for episode in buf.episodes]}")

'''
FilePath: /torch-rl/agent.py
Date: 2022-08-31 18:11:42
LastEditTime: 2022-09-10 11:20:35
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
import torch.distributions as D

import gym
from gym import spaces

from magic_rl.buffers.buffer import ReplayBuffer
from magic_rl.networks.network import CriticNetwork, PolicyNetwork


class SacAgent(object):
    '''
    Soft Actor-Critic version 2
    using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
    add alpha loss compared with version 1
    paper: https://arxiv.org/pdf/1812.05905.pdf
    '''

    def __init__(self,
        state_dims,
        action_dims,
        is_auto_entropy:bool=False,
        chkpt_dir:str="chkpt/",
        device=torch.device("cpu"),
        ) -> None:
        # attributes
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.chkpt_dir = chkpt_dir
        self.device = device
        self.is_auto_entropy = is_auto_entropy
        
        self.gamma = 0.99
        self.soft_tau = 1e-2

        self.action_scale = 1
        self.reward_scale = 10

        # networks
        self.critic_net_1 = CriticNetwork("critic_net_1.pth", self.chkpt_dir, self.state_dims, self.action_dims, optim.Adam, 3e-4, self.device)
        self.critic_net_2 = CriticNetwork("critic_net_2.pth", self.chkpt_dir, self.state_dims, self.action_dims, optim.Adam, 3e-4, self.device)
        self.target_critic_net_1 = CriticNetwork("target_critic_net_1.pth", self.chkpt_dir, self.state_dims, self.action_dims, optim.Adam, 3e-4, self.device)
        self.target_critic_net_2 = CriticNetwork("target_critic_net_2.pth", self.chkpt_dir, self.state_dims, self.action_dims, optim.Adam, 3e-4, self.device)
        self.policy_net = PolicyNetwork("policy_net.pth", self.chkpt_dir, self.state_dims, self.action_dims, optim.Adam, 3e-4, self.device)

        for target_param, param in zip(self.target_critic_net_1.parameters(), self.critic_net_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_net_2.parameters(), self.critic_net_2.parameters()):
            target_param.data.copy_(param.data)
    
    def save_chkpt(self):
        self.critic_net_1.save_chkpt()
        self.critic_net_2.save_chkpt()
        self.policy_net.save_chkpt()

    def load_chkpt(self, chkpt_file:str=None):
        self.critic_net_1.load_chkpt()
        self.critic_net_2.load_chkpt()
        self.policy_net.load_chkpt()

    def sample_action(self, state, is_deterministic:bool=False):
        '''
        sample one action under given state
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        mean, log_std = self.policy_net(state)

        if is_deterministic:
            action = torch.tanh(mean)  # tanh function plays an important role here.
        else:
            std = log_std.exp()
            samples = D.Normal(0, 1).sample(mean.shape).to(self.device)  # samples from standard gaussian distribution
            action = torch.tanh(mean + std * samples)

        return self.action_scale * action.detach().cpu().numpy()[0]

    def evaluate(self, state):
        '''
        sample one action under given state
        '''
        mean, log_std = self.policy_net(state)
        std = log_std.exp()
        
        samples = D.Normal(0, 1).sample(mean.shape)  # samples from standard gaussian distribution
        action_normal = torch.tanh(mean + std * samples.to(self.device))
        action = self.action_scale * action_normal
        log_prob = D.Normal(mean, std).log_prob(mean + std * samples.to(self.device)) - torch.log(1. - action_normal.pow(2) + 1e-6) - np.log(self.action_scale)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def update(self, samples):
        '''
        sample memories from buffer, update network several times
        '''
        obs, action, reward, next_obs, done = samples

        # from numpy to tensor
        obs      = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action   = torch.FloatTensor(action).to(self.device)
        reward   = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done     = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.critic_net_1(obs, action)
        predicted_q_value2 = self.critic_net_2(obs, action)
        new_action, log_prob = self.evaluate(obs)
        new_next_action, next_log_prob = self.evaluate(next_obs)
        reward = self.reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if self.is_auto_entropy:
            pass
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_critic_net_1(next_obs, new_next_action), self.target_critic_net_2(next_obs, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.gamma * target_q_min
        q_value_loss1 = nn.MSELoss()(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = nn.MSELoss()(predicted_q_value2, target_q_value.detach())

        self.critic_net_1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.critic_net_1.optimizer.step()        
        self.critic_net_2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.critic_net_2.optimizer.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.critic_net_1(obs, new_action), self.critic_net_2(obs, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_critic_net_1.parameters(), self.critic_net_1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_critic_net_2.parameters(), self.critic_net_2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
            
        return q_value_loss1, q_value_loss2, policy_loss


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


if __name__ == "__main__":
    # choose env
    env = NormalizedActions(gym.make("Pendulum-v1"))
    obs_dims = env.observation_space.shape[0]
    act_dims = env.action_space.shape[0]

    # replay buffer
    replay_buffer = ReplayBuffer(1e6)

    # sac agent
    sac_agent = SacAgent(
        state_dims=obs_dims,
        action_dims=act_dims,
        is_auto_entropy=False,
        chkpt_dir="chkpt/",
        device=torch.device("cuda:0"),
    )

    # logger
    from torch.utils.tensorboard import SummaryWriter
    sac_logger = SummaryWriter(log_dir="outputs/sac_v2/log/")

    # hyper-parameters for RL training
    max_episodes = 1_000_000
    max_steps = 150

    # others
    frame_idx = 0
    batch_size = 300
    update_itr = 1
    
    for eps in range(max_episodes):
        episode_lens = 0
        episode_reward = 0

        obs = env.reset()
        for step in range(max_steps):
            action = sac_agent.sample_action(obs)
            next_obs, reward, done, _ = env.step(action)
            
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            frame_idx += 1

            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    q_value_loss1, q_value_loss2, policy_loss = sac_agent.update(replay_buffer.sample(batch_size))
                    sac_logger.add_scalar("q_value_loss1", q_value_loss1, eps)
                    sac_logger.add_scalar("q_value_loss2", q_value_loss2, eps)
                    sac_logger.add_scalar("policy_loss", policy_loss, eps)

            if done:
                break
        
        print('Episode: ', eps, '| Episode Reward: ', episode_reward)
        sac_logger.add_scalar("episode reward", episode_reward, eps)
        
        if eps % 20 == 0 and eps > 0:
            sac_agent.save_chkpt()
    
    sac_agent.save_chkpt()
    
