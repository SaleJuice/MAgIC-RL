'''
FilePath: /MAgIC-RL/utils.py
Date: 2022-09-07 13:28:25
LastEditTime: 2022-09-10 22:02:13
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''


import numpy as np

import gym

class LinearAnneal:
    """Linear Annealing Schedule.

    Args:
        start: The initial value of epsilon.
        end: The final value of epsilon.
        duration: The number of anneals from start value to end value.

    """

    def __init__(self, start: float, end: float, duration: int):
        self.val = start
        self.min = end
        self.duration = duration

    def anneal(self):
        self.val = max(self.min, self.val - (self.val - self.min) / self.duration)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action


if __name__ == "__main__":
    pass