'''
FilePath: /MAgIC-RL/magic_rl/utils/logger_utils.py
Date: 2022-09-13 18:26:15
LastEditTime: 2022-09-15 15:58:08
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

import abc

import wandb


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


if __name__ == "__main__":
    import random
    import time

    logger = WandbLogger(project="new-project", group="experiment_3", job_type="train")

    logger.config({"episode":63, "batch_size": 256})
    for i in range(120):
        logger.log({"loss/v": random.randint(0, 100)})
        logger.log({"loss/c": random.randint(0, 100)})
        time.sleep(0.5)

    logger.finish()
        
