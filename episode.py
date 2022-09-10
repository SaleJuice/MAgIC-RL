'''
FilePath: /MAgIC-RL/episode.py
Date: 2022-08-31 09:38:30
LastEditTime: 2022-09-10 22:01:58
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

from typing import Dict, List
import numpy as np


class Episode(object):
    '''
    Attention: this 'Episode' contains the final observation.
    '''

    def __init__(self) -> None:
        self.obss: List[np.ndarray] = []
        self.acts: List[np.ndarray] = []
        self.rews: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []

    def __len__(self) -> int:
        return self.step_nums

    def add(self, obs, act, rew, done) -> None:
        assert (isinstance(obs, (np.ndarray))), f"Error on type of 'observation' data!"
        assert (isinstance(act, (int, type(None)))), f"Error on type of 'action' data!"
        assert (isinstance(rew, (float, type(None)))), f"Error on type of 'reward' data!"
        assert (isinstance(done, (bool, type(None)))), f"Error on type of 'done' data!"

        self.obss.append(obs)
        self.acts.append(np.array([act]))
        self.rews.append(np.array([rew]))

        if done is True:
            self.dones.append(np.array([1]))
        elif done is None:
            self.dones.append(np.array([None]))
        else:
            self.dones.append(np.array([0]))

    def get(self, index):
        assert (index < self.step_nums), f"Index '{index}' is out of range."

        obs = self.obss[index]
        act = self.acts[index]
        rew = self.rews[index]
        done = self.dones[index]

        return obs, act, rew, done

    def sample(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        assert (self.is_samplable), f"There is no step can be sampled."

        batch = {}
        batch["obs"] = []
        batch["act"] = []
        batch["rew"] = []
        batch["done"] = []
        batch["obs_next"] = []

        indexs = np.random.choice(self.step_nums - 1, size=min(batch_size, self.step_nums - 1), replace=False)

        for i in indexs:
            obs, act, rew, done = self.get(i)
            obs_next, act_next, rew_next, done_next = self.get(i + 1)
            batch["obs"].append(obs)
            batch["act"].append(act)
            batch["rew"].append(rew)
            batch["done"].append(done)
            batch["obs_next"].append(obs_next)

        batch["obs"] = np.array(batch["obs"])
        batch["act"] = np.array(batch["act"])
        batch["rew"] = np.array(batch["rew"])
        batch["done"] = np.array(batch["done"])
        batch["obs_next"] = np.array(batch["obs_next"])

        return batch

    @property
    def is_samplable(self) -> bool:
        return True if self.step_nums > 0 else False

    @property
    def is_complete(self) -> bool:
        return True if None in self.dones else False

    @property
    def step_nums(self) -> int:
        return len(self.obss)

    @property
    def cumulative_rew(self):
        if self.is_complete:
            return sum(self.rews[:-1])
        else:
            return sum(self.rews[:])


if __name__ == "__main__":
    import gym

    epi = Episode()  # <-- creat

    print(f"")
    print(f"length: {len(epi)}")
    print(f"is_samplable: {epi.is_samplable}")
    print(f"is_complete: {epi.is_complete}")
    print(f"cumulative_rew: {epi.cumulative_rew}")

    env = gym.make("CartPole-v1")
    obs = env.reset()
    while True:
        act = env.action_space.sample()
        obs_next, rew, done, info = env.step(act)
        epi.add(obs, act, rew, done)  # <-- add
        obs = obs_next

        print(f"")
        print(f"length: {len(epi)}")
        print(f"is_samplable: {epi.is_samplable}")
        print(f"is_complete: {epi.is_complete}")
        print(f"cumulative_rew: {epi.cumulative_rew}")

        if done:
            epi.add(obs, None, None, None)  # <-- add
            break

    print(f"")
    print(f"length: {len(epi)}")
    print(f"is_samplable: {epi.is_samplable}")
    print(f"is_complete: {epi.is_complete}")
    print(f"cumulative_rew: {epi.cumulative_rew}")

    print(f"sample: {epi.sample(10)}")  # <-- sample
