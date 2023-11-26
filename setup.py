'''
FilePath: /MAgIC-RL/setup.py
Date: 2022-09-13 12:32:37
LastEditTime: 2022-09-16 20:41:57
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

from setuptools import setup


setup(
    name='magic_rl',
    version='0.0.1',
    install_requires=['numpy', 'torch', 'tensorboard', 'gymnasium', 'wandb', 'prettytable']
)
