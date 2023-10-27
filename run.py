'''
FilePath: /MAgIC-RL/run.py
Date: 2022-09-13 12:45:42
LastEditTime: 2023-10-26 21:08:43
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import argparse

import torch

import gym
# import pybullet_envs  # noqa

from magic_rl.buffers.buffer import ReplayBuffer
from magic_rl.agents.sac_agent import SacAgent
from magic_rl.schedulers.trainer import Trainer
from magic_rl.schedulers.evaluator import Evaluator
from magic_rl.utils.logger_utils import WandbLogger, TensorboardLogger
from magic_rl.utils.gym_utils import NormalizeActions, wrap_env
from magic_rl.utils.others_utils import formate_args_as_table


def parse_args():
    parser = argparse.ArgumentParser()

    # job type related:
    # -----------------
    parser.add_argument("--job-type", type=str, default="eval", choices=['train', 'eval', 'retrain'],
                        help="")
    
    # environment related:
    # --------------------
    parser.add_argument("--train-env", type=str, default="gym_fish:Sim2Real-v0", choices=['gym_fish:Sim2Real-v0', 'LunarLanderContinuous-v2', 'Pendulum-v0'],
                        help="The environment be selected for training a rl agent.")

    parser.add_argument("--eval-env", type=str, default="gym_fish:Sim2Real-v0",  choices=['gym_fish:Sim2Real-v0', 'LunarLanderContinuous-v2', 'Pendulum-v0'],
                        help="The environment be selected for evaluating a rl agent.")
    
    parser.add_argument("--render", type=bool, default=False, choices=['True', 'False'],
                        help="Whether to render the environment.")

    # FIXME abandoned
    # parser.add_argument("--env-wrappers", type=str, default="NormalizeActions", 
    #                     help="Using environment wrapper, you can change environment feature easily.")
    
    # replay buffer related:
    # ----------------------
    parser.add_argument("--buffer-size", type=str, default=1e6, 
                        help="The size of replay buffer for off-policy rl agent.")

    parser.add_argument("--start-steps", type=str, default=5e3, 
                        help="")

    # rl agent related:
    # -----------------
    parser.add_argument("--agent", type=str, default="sac_agent", choices=['sac_agent'],
                        help="The rl agent (not specified means random action) for training or evaluating in enviroments.")

    parser.add_argument("--wb-dir", type=str, default="./checkpoints/8e588b5e/steps_7414", 
                        help="If you want to evaluate or retrain a agent with exist weight and bias.")

    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda'],
                        help="The device that rl algorithm run with.")
    
    # logger related:
    # ---------------
    parser.add_argument("--logger", type=str, default="tensorboard", choices=['tensorboard', 'wandb'],
                        help="The type of logger you want to use.")

    parser.add_argument("--project-name", type=str, default="demo-project", 
                        help="The project name for logger to classify different runs.")

    parser.add_argument("--group-name", type=str, default="demo-group", 
                        help="The group name for logger to classify different runs.")

    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="'0' means no printing log; '1' means some necessary printing log; '2' means all printing log.")

    # scheduler related:
    # ------------------
    parser.add_argument("--experiment-length", type=str, default="episodes:1e3", 
                        help="The length of experiment for training or evaluating. ('episodes:1e3', 'steps:1e6')")
    
    return parser.parse_args()


def run_experiment(args):
    # initialize logger for logging any thing you need:
    if args.logger is None:
        logger = None
    else:
        if "wandb" in args.logger:
            logger = WandbLogger(project=args.project_name, group=args.group_name, job_type=args.job_type)
        elif "tensorboard" in args.logger:
            logger = TensorboardLogger(project=args.project_name, group=args.group_name, job_type=args.job_type)
        else:
            assert (False), f"Do NOT support this '{args.logger}' logger!"
        logger.config(args)

    # what kind of job do you want to do
    if "train" in args.job_type:
        # initialize environment:
        assert (args.train_env is not None), f"Please specify the '--train-env', if you want to do training related task."
        
        # FIXME delete the wrapper function or make it more useful.
        # train_env = wrap_env(gym.make(args.train_env), args.env_wrappers.split(":"))
        train_env = gym.make(args.train_env)
        
        # TODO automatically distinguish between discrete and continuous environments.
        observation_dim  = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        
        if args.eval_env is not None:
            eval_env = gym.make(args.eval_env)
        else:
            eval_env = gym.make(args.train_env)

        # initialize rl agent:
        assert (args.agent is not None), f"Please specify the '--agent', if you want to train the rl agent."
        if "sac_agent" in args.agent:
            agent = SacAgent(state_dim=observation_dim, action_dim=action_dim, hidden_dim=256, action_range=1.0, device=torch.device(args.device))
        else:
            assert (False), f"Do not support this '{args.agent}' agent yet, please choose another agent!"

        # initialize replay buffer:
        buffer = ReplayBuffer(int(float(args.buffer_size)))

        # initialize scheduler and create an schedule which the scheduler running experiment according to:
        scheduler = Trainer(env=train_env, agent=agent, buffer=buffer, logger=logger, verbose=args.verbose)

        schedule = {}
        schedule[args.experiment_length.split(":")[0]] = int(float(args.experiment_length.split(":")[1]))
        schedule["render"] = args.render
        schedule["batch_size"] = 256
        schedule["start_steps"] = 5e3
        schedule["save_model_interval"] = 10

        # do some more operations if you want to 'retrain':
        if "retrain" in args.job_type:
            assert (args.wb_dir is not None), f"Please specify the '--wb-file', if you want to retrain the agent with exist weight and bias."
            agent.load_model(args.wb_dir)

    elif "eval" in args.job_type:
        # initialize environment:
        assert (args.eval_env is not None), f"Please specify the '--eval-env', if you want to do evaluating related task."
        eval_env = gym.make(args.eval_env)
        
        # TODO automatically distinguish between discrete and continuous environments.
        observation_dim  = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.shape[0]

        # initialize rl agent:
        device = torch.device(args.device)
        
        if args.agent is None:  # means agent with random action.
            agent = None
        else:
            assert (args.wb_dir is not None), f"Please specify the '--wb-file', if you want to evaluate the agent with exist wb(weight and bias)."
            if "sac_agent" in args.agent:
                agent = SacAgent(state_dim=observation_dim, action_dim=action_dim, hidden_dim=256, action_range=1.0, device=device)
            else:
                assert (False), f"Do not support this '{args.agent}' agent yet, please choose another agent!"
            agent.load_model(args.wb_dir)

        # initialize scheduler and create an schedule which the scheduler running experiment according to:
        scheduler = Evaluator(env=eval_env, agent=agent, logger=logger, render=args.render, verbose=args.verbose)

        schedule = {}
        schedule[args.experiment_length.split(":")[0]] = int(float(args.experiment_length.split(":")[1]))

    else:
        assert (False), f"Do not support this '{args.job_type}' job yet, please choose another job!"

    # running the scheduler according to schedule now:
    scheduler.run(schedule=schedule)

    # closing for ending:
    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    args = parse_args()

    # check the arguments got and wait to run the experiment:
    # -------------------------------------------------------
    print(formate_args_as_table(args))
    while True:
        answer = input(f"Do you want to continue? (yes/no) :")
        if answer == "yes":
            run_experiment(args)
            break
        elif answer == "no":
            break
