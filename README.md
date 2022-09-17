# MAgIC-RL

Simple RL algorithms realized by pytorch.

## MAgIC-RL
```
--train-env Pendulum-v0 --job-type train --agent sac_agent --buffer-size 1e6 --device cuda:0 --logger wandb --project-name MAgIC-RL --group-name Pendulum-v0 --experiment-length steps:1e6
```

## GymFish
### train
```
--train-env GymFish-v1 --job-type train --agent sac_agent --buffer-size 1e6 --device cuda:0 --logger wandb --project-name Gym-Fish --group-name v1 --experiment-length steps:1e5
```
### evaluate
```
--eval-env GymFish-v1 --job-type eval --agent sac_agent --wb-dir checkpoints/5y8xcwpa/steps_99947/ --device cuda:0 --logger wandb --project-name Gym-Fish --group-name v1 --experiment-length episodes:10
```

## Try
```
--train-env GymFish-v2 --job-type train --agent sac_agent --buffer-size 1e6 --device cuda:0 --logger wandb --project-name Gym-Fish --group-name v2 --experiment-length steps:1e5
```
