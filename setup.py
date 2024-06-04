from setuptools import setup, find_packages


setup(
    name='magic_rl',
    version='0.1.0',
    author='Xiaozhu Lin',
    author_email='linxzh@shanghaitech.edu.cn',
    description='A lightweight, concise, and research friendly Reinforcement Learning (RL) algorithm implementation based on PyTorch.',
    url='https://github.com/SaleJuice/MAgIC-RL',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy', 'torch', 'tensorboard', 'gymnasium', 'wandb', 'prettytable']
)
