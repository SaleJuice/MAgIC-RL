'''
FilePath: /MAgIC-RL/magic_rl/utils/logger_utils.py
Date: 2022-09-13 18:26:15
LastEditTime: 2023-10-24 14:57:10
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import abc
from typing import Union
import os
import json
import time
import argparse

import wandb
from torch.utils.tensorboard import SummaryWriter

import uuid


class Logger(object):
    '''
    An abstract base class for different types of logger, such as 'wandb', 'tensorboard', etc.
    '''
    
    @abc.abstractmethod
    def __init__(self, project:str, group:str, job_type:str, **kwargs) -> None:
        '''
        To provide classification information for each running.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def config(self, *args):
        '''
        To record some configuration information, such as hyperparameters, environments, tasks, etc.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, *args):
        '''
        To record once information, such as loss, accuracy, etc.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self):
        '''
        To close the logger for some reason.
        '''
        raise NotImplementedError


class WandbLogger(Logger):
    '''
    Because of the limitation of 'wandb', this class can only be instantiated one at a time.
    '''

    def __init__(self, project:str, group:str, job_type:str, reinit:bool=True, **kwargs) -> None:
        self.project = project
        self.group = group
        self.job_type = job_type
        self.reinit = reinit

        self.run = wandb.init(project=self.project, group=self.group, job_type=self.job_type, reinit=self.reinit, **kwargs)

    @property
    def id(self):
        return wandb.run.id
    
    def config(self, *args):
        wandb.config.update(*args)

    def log(self, *args):
        wandb.log(*args)

    def finish(self):
        self.run.finish()


class TensorboardLogger(Logger):
    '''
    for local logging.
    '''
    def __init__(self, project:str, group:str, job_type:str, **kwargs) -> None:
        self.project = project
        self.group = group
        self.job_type = job_type

        self.id = str(uuid.uuid1()).split("-")[0]
        self.start_timestamp = time.time()

        self.run_name = f"run-{time.strftime('%Y%m%d_%H%M%S', time.localtime(self.start_timestamp))}-{self.id}"
        self.files_dir = os.path.join("./", "tensorboard/", f"{self.run_name}/", "files/")
        self.logs_dir = os.path.join("./", "tensorboard/", f"{self.run_name}/", "logs/")
        self.tmp_dir = os.path.join("./", "tensorboard/", f"{self.run_name}/", "tmp/")

        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.run = SummaryWriter(log_dir=self.logs_dir)
        self.run_steps = 0

        # save some information as json file
        meta_data = {}
        meta_data["id"] = self.id
        meta_data["start_timestamp"] = self.start_timestamp
        meta_data["project"] = self.project
        meta_data["group"] = self.group
        meta_data["job_type"] = self.job_type

        with open(os.path.join(self.files_dir, "metadata.json"), "w") as fp:
            json.dump(meta_data, fp, indent=4)

        self.config_data = {}
    
    def config(self, *args:Union[dict, argparse.Namespace]):
        if isinstance(*args, dict):
            self.config_data.update(*args)
        elif isinstance(*args, argparse.Namespace):  # TODO check if it works or not
            for arg in vars(*args):
                self.config_data[arg] = getattr(*args, arg)
        
        with open(os.path.join(self.files_dir, "config.json"), "w") as fp:
            json.dump(self.config_data, fp, indent=4)

    def log(self, *args:Union[dict, None]):
        if isinstance(*args, dict):
            for key, value in args[0].items():
                self.run.add_scalar(key, value, self.run_steps)
            self.run_steps += 1
        else:
            pass

    def finish(self):
        pass
    

if __name__ == "__main__":
    import random
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])

    args = parser.parse_args()

    if "wandb" in args.logger:
        logger = WandbLogger(project="new-project", group="experiment_3", job_type="train")
    elif "tensorboard" in args.logger:
        logger = TensorboardLogger(project="new-project", group="experiment_3", job_type="train")
    else:
        assert (False), f"The '{args.logger}' logger is not supported yet."

    logger.config(args)
    logger.config({"episode":63, "batch_size": 256})
    
    for i in range(1000):
        logger.log(
            {
                "loss/c": random.randint(0, 100),
                "loss/v": random.randint(0, 100),
            }
        )
        time.sleep(0.001)

    logger.finish()
        
