import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity

        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.position = 0

    def push(self, obs, act, rew, next_obs, ter, tru):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, act, rew, next_obs, ter, tru)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, ter, tru = map(np.stack, zip(*batch))  # stack for each element
        return obs, act, rew, next_obs, ter, tru

    def pour(self):
        obs, act, rew, next_obs, ter, tru = map(np.stack, zip(*self.buffer))  # stack for each element
        return obs, act, rew, next_obs, ter, tru


if __name__ == "__main__":
    replay_buffer = ReplayBuffer(capacity=100)
