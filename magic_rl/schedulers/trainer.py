'''
FilePath: /MAgIC-RL/magic_rl/schedulers/trainer.py
Date: 2022-09-13 21:39:42
LastEditTime: 2022-09-14 12:20:08
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

from typing import Dict, Union

import wandb

import gym

# TODO from magic_rl.agents.agent import Agent
from magic_rl.utils.logger_utils import Logger


class Trainer(object):
    '''
    To evaluate the performance of agent with trained model or the performance of random actions.
    '''

    def __init__(self, env:gym.Env, agent, buffer, logger:Logger, render:bool=False, verbose:int=1) -> None:
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.render = render
        self.verbose = verbose

    def run(self, schedule:Dict[str, int]):
        assert (len(schedule) == 1), f"Expect a 'dict' type variable which length is '1', but get '{len(schedule)}'!"
        assert ("step" in schedule.keys() or "episode" in schedule.keys()), f"Illegal schedule '{schedule}' of which the key have to be one of 'episode' and 'step'."

        steps = 0
        episodes = 0
        q_value_loss1, q_value_loss2, policy_loss, alpha_loss = 0, 0, 0, 0

        while True:  # episode
            episode_len = 0
            episode_rew = 0

            obs = self.env.reset()
            while True:  # step  # XXX limit num of steps in one episode if possable.
                if self.agent is not None:
                    act = self.agent.get_action(obs, deterministic = False)  # TODO 
                else:
                    assert (False), f"It is illegal to train agent without specifying agent type."
                
                next_obs, rew, done, _ = self.env.step(act)

                self.buffer.push(obs, act, rew, next_obs, done)
                obs = next_obs
                
                if self.render:
                    self.env.render()
                
                steps += 1
                episode_len += 1
                episode_rew += rew

                if len(self.buffer) > 256:
                    batch_buff = self.buffer.sample(256)
                    q_value_loss1, q_value_loss2, policy_loss, alpha_loss = self.agent.update(batch_buff, reward_scale=20., auto_entropy=True, target_entropy=-4)
                        
                if done:
                    break

                if "step" in schedule.keys() and steps >= schedule["step"]:  # overflow check 
                    break
            
            if self.verbose > 0:
                print(f"Steps: {steps} | Episodes: {episodes} | Episode Length: {episode_len} | Episode Reward: {episode_rew}")
            
            if self.logger is not None:
                self.logger.log(
                    {
                        "train/episode_len":episode_len,
                        "train/episode_rew":episode_rew,
                        "loss/q_value_loss1":q_value_loss1,
                        "loss/q_value_loss2":q_value_loss2,
                        "loss/policy_loss":policy_loss,
                        "loss/alpha_loss":alpha_loss,
                    }
                )
            
            episodes += 1

            if "episode" in schedule.keys() and episodes >= schedule["episode"]:  # overflow check 
                break
            
            if "step" in schedule.keys() and steps >= schedule["step"]:  # overflow check 
                    break



if __name__ == '__main__':
    # init
    env = gym.make("Pendulum-v0")
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    from magic_rl.buffers.buffer import ReplayBuffer
    buffer = ReplayBuffer(1e6)
    
    import torch
    from magic_rl.agents.sac_agent import SacAgent
    agent = SacAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, action_range=1.0, device=torch.device("cuda:0"))
    
    from magic_rl.utils.logger_utils import WandbLogger
    logger = WandbLogger(project="demo-project", group="exp-1", job_type="train")
    
    render = False
    verbose = 1

    # create
    scheduler = Trainer(env=env, agent=agent, buffer=buffer, logger=logger, render=render, verbose=verbose)
    
    # run now!
    scheduler.run({"episode": 1e3})
