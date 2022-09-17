'''
FilePath: /MAgIC-RL/magic_rl/utils/gym_utils.py
Date: 2022-09-14 14:30:18
LastEditTime: 2022-09-17 18:42:20
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import numpy as np

import gym
from gym import spaces


class NormalizeActions(gym.ActionWrapper):
    '''
    This class is used to normalize the action space of the original env (must be continuous action env).
    '''
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(self.action_space, spaces.Box), f"Must be continuous action env!"

        self.ori_low = self.action_space.low
        self.ori_high = self.action_space.high
        
        high = np.ones(self.action_space.shape, dtype=np.float32)
        low = - high
        self.action_space = spaces.Box(low=low, high=high)

    def action(self, action):
        action = self.ori_low + (action + 1.0) * 0.5 * (self.ori_high - self.ori_low)
        action = np.clip(action, self.ori_low, self.ori_high)
        
        return action

    def reverse_action(self, action):
        action = 2 * (action - self.ori_low) / (self.ori_high - self.ori_low) - 1
        action = np.clip(action, self.ori_low, self.ori_high)
        
        return action


class SequenceObservations(gym.ObservationWrapper):
    '''
    This class is used to sequence the observation of the original env (must be continuous observation env).
    '''
    
    def __init__(self, env: gym.Env, sequence_len:int=3) -> None:
        super().__init__(env)
        self.sequence_len = sequence_len

        self.ori_low = self.observation_space.low
        self.ori_high = self.observation_space.high

        high = np.tile(self.ori_high, self.sequence_len)
        low = np.tile(self.ori_low, self.sequence_len)
        # XXX ori_low/ori_high have to be array only
        self.observation_space = spaces.Box(low=low, high=high)

        self.observation_sequence = np.zeros(self.observation_space.shape)
    
    def observation(self, observation):
        self.observation_sequence = np.concatenate((self.observation_sequence[len(observation):], observation), axis=0)
        return self.observation_sequence

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.observation_sequence = np.tile(observation, self.sequence_len)
        return self.observation_sequence


class AddActionsToObservations(gym.ObservationWrapper):
    '''
    This class is used to add the past actions to observations.
    '''
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.ori_obs_low = self.observation_space.low
        self.ori_obs_high = self.observation_space.high

        self.ori_act_low = self.action_space.low
        self.ori_act_high = self.action_space.high

        high = np.concatenate((self.ori_obs_high, self.ori_act_high), axis=0)
        low = np.concatenate((self.ori_obs_low, self.ori_act_low), axis=0)
        self.observation_space = spaces.Box(low=low, high=high)
    
    def observation(self, observation, action):
        observation = np.concatenate((observation, action), axis=0)
        return observation

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        action = np.zeros(self.action_space.sample().shape)
        return self.observation(observation, action)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation, action), reward, done, info


if __name__ == '__main__':
    # Original Environment
    print("Original Environment:")
    ori_env = gym.make("Pendulum-v0")
    print(ori_env.observation_space, ori_env.action_space)

    # NormalizeActions
    print("NormalizeActions:")
    na_env = NormalizeActions(ori_env)
    print(na_env.observation_space, na_env.action_space)

    # SequenceObservations
    print("SequenceObservations:")
    so_env = SequenceObservations(ori_env)
    print(so_env.observation_space, so_env.action_space)

    # AddActionsToObservations
    print("AddActionsToObservations:")
    aato_env = AddActionsToObservations(ori_env)
    print(aato_env.observation_space, aato_env.action_space)

    # AddActionsToObservations + SequenceObservations
    print("AddActionsToObservations + SequenceObservations:")
    aato_so_env = SequenceObservations(AddActionsToObservations(ori_env))
    print(aato_so_env.observation_space, aato_so_env.action_space)

    # exit()
    env = aato_so_env
    obs = env.reset()
    
    all_rew = 0
    while True:
        print("")
        print(f"====================")

        # env.render()
        act = env.action_space.sample()
        next_obs, rew, done, info = env.step(act)
        all_rew += rew

        print(obs)
        print(act)
        print(rew)
        print(next_obs)

        obs = next_obs

        if done:
            print(all_rew)
            break
