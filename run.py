'''
FilePath: /MAgIC-RL/run.py
Date: 2022-09-13 12:45:42
LastEditTime: 2022-09-13 21:37:33
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import argparse

import wandb

import gym

from magic_rl.buffers.buffer import ReplayBuffer
from magic_rl.agents.sac_agent import SacAgent
from magic_rl.schedulers.evaluator import Evaluator
from magic_rl.utils.logger_utils import WandbLogger


def run_experiment(args):
    # init environment:
    # -----------------
    train_env = gym.make(args.train_env)
    eval_env = gym.make(args.eval_env)

    # init replay buffer:
    # -------------------
    buffer = ReplayBuffer(args.buffer_size)

    # init rl agent:
    # --------------
    if args.agent == None:  # means agent with random action.
        agent = None
    elif args.agent == "sac_agent":
        agent = SacAgent()
    else:
        assert (False), f"Do not support this '{args.job_type}' agent yet, please choose another agent!"

    # init logger:
    # ------------
    if args.logger == None:  # means do not need any logger for logging.
        logger = None
    elif args.logger == "wandb":
        logger = WandbLogger(project=args.project_name, group=args.group_name, job_type=args.job_type)
        logger.config(args)
    elif args.logger == "tensorboard":
        logger = None  # TODO TensorboardLogger(project=args.project_name, group=args.group_name, job_type=args.job_type)
    else:
        assert (False), f"Do not support this '{args.logger}' logger yet, please choose another logger!"

    # init scheduler:
    # ---------------
    if args.job_type == "train":
        scheduler = None  # TODO Trainer(train_env=train_env, eval_env=eval_env, buffer=buffer, agent=agent, logger=logger)
    elif args.job_type == "eval":
        scheduler = Evaluator(env=eval_env, agent=agent, logger=logger, render=args.render, verbose=args.verbose)
    else:
        assert (False), f"Do not support this '{args.job_type}' job yet, please choose another job!"

    # running now:
    # ------------
    assert ((args.num_episodes is not None and args.num_steps is None) or (args.num_episodes is None and args.num_steps is not None)), f"Please just only choose one argument of '--num-episodes' and '--num-steps' to fill."
    
    if args.num_episodes is not None:
        scheduler.run({"episode":args.num_episodes})
    elif args.num_steps is not None:
        scheduler.run({"step":args.num_steps})
    else:
        assert (False), f"Please at least choose one argument of '--num-episodes' and '--num-steps' to fill."

    # close for ending:
    # -----------------
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # environment related:
    # --------------------
    parser.add_argument(
        "--train-env",
        type=str,
        default="Pendulum-v0",
        help="The environment be selected for training a rl agent.",
    )

    parser.add_argument(
        "--eval-env",
        type=str,
        default="Pendulum-v0",
        help="The environment be selected for evaluating a rl agent.",
    )

    parser.add_argument(
        "--render",
        type=bool,
        default=False,
        help="The flag that determines whether to render the environment.",
    )
    
    # replay buffer related:
    # ----------------------
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1e6,
        help="The size of replay buffer for off-policy rl agent.",
    )

    # rl agent related:
    # -----------------
    parser.add_argument(
        "--agent",
        type=str,
        default=None,  # "sac_agnet"
        help="The rl agent for training or evaluating in enviroments. ('None' means agent with random action)",
    )
    
    # logger related:
    # ---------------
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        help="The type of logger you want to use, 'wandb' for online or 'tensorboard' for local.",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="unknown-project",
        help="The project name for logger to classify different runs.",
    )

    parser.add_argument(
        "--group-name",
        type=str,
        default="unknown-group",
        help="The group name for logger to classify different runs.",
    )

    parser.add_argument(
        "--job-type",
        type=str,
        default="eval",  # "eval", "train"
        help="The job type for logger to classify different runs.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="'0' means no printing log; '1' means some necessary printing log; '2' means all printing log.",
    )

    # scheduler related:
    # ------------------
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1e2,  # 1e2
        help="The number of episodes for training or evaluating. (only choose one of '--num-episodes' and '--num-steps')",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,  # 1e5
        help="The number of steps for training or evaluating. (only choose one of '--num-episodes' and '--num-steps')",
    )


    run_experiment(parser.parse_args())
