'''
FilePath: /MAgIC-RL/magic_rl/schedulers/evaluator.py
Date: 2022-09-13 15:58:39
LastEditTime: 2022-09-13 21:13:42
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


class Evaluator(object):
    '''
    To evaluate the performance of agent with trained model or the performance of random actions.
    '''

    def __init__(self, env:gym.Env, agent, logger:Logger, render:bool=False, verbose:int=1) -> None:
        self.env = env
        self.agent = agent
        self.logger = logger
        self.render = render
        self.verbose = verbose

    def run(self, schedule:Dict[str, int]):
        assert (len(schedule) == 1), f"Expect a 'dict' type variable which length is '1', but get '{len(schedule)}'!"
        assert ("step" in schedule.keys() or "episode" in schedule.keys()), f"Illegal schedule '{schedule}' of which the key have to be one of 'episode' and 'step'."

        steps = 0
        episodes = 0

        while True:  # episode
            episode_len = 0
            episode_rew = 0

            obs = self.env.reset()
            while True:  # step  # XXX limit num of steps in one episode if possable.
                if self.agent is not None:
                    act = self.agent.sample_action(obs)  # TODO 
                else:
                    act = self.env.action_space.sample()
                
                obs, rew, done, _ = self.env.step(act)
                
                if self.render:
                    self.env.render()
                
                steps += 1
                episode_len += 1
                episode_rew += rew
                        
                if done:
                    break

                if "step" in schedule.keys() and steps >= schedule["step"]:  # overflow check 
                    break
            
            if self.verbose > 0:
                print(f"Steps: {steps} | Episodes: {episodes} | Episode Length: {episode_len} | Episode Reward: {episode_rew}")
            
            if self.logger is not None:
                self.logger.log(
                    {
                        "eval/episode_len":episode_len,
                        "eval/episode_rew":episode_rew,
                    }
                )
            
            episodes += 1

            if "episode" in schedule.keys() and episodes >= schedule["episode"]:  # overflow check 
                break
            
            if "step" in schedule.keys() and steps >= schedule["step"]:  # overflow check 
                    break


if __name__ == "__main__":
    # init
    env = gym.make("Pendulum-v0")
    agent = None
    
    from magic_rl.utils.logger_utils import WandbLogger
    logger = WandbLogger(project="demo-project", group="exp-1", job_type="eval")
    
    render = False
    verbose = 1

    # create
    scheduler = Evaluator(env=env, agent=agent, logger=logger, render=render, verbose=verbose)
    
    # run now!
    scheduler.run({"episode": 1e3})