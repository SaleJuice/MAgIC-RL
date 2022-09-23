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
--eval-env GymFish-v1 --job-type eval --agent sac_agent --wb-dir checkpoints/3exochh8/steps_49656/ --device cuda:0 --logger wandb --project-name Gym-Fish --group-name v1 --experiment-length episodes:10
--eval-env GymFish-v1 --job-type eval --agent sac_agent --wb-dir checkpoints/1xme5own/steps_83587/ --device cuda:0 --logger wandb --project-name Gym-Fish --group-name v1 --experiment-length episodes:10
```

## Try
```
--train-env GymFish-v4 --env-wrappers SequenceObservations:AddActionsToObservations:NormalizeActions --job-type train --agent sac_agent --buffer-size 1e6 --device cuda:0 --logger wandb --project-name Gym-Fish --group-name v4:so:aato:na --experiment-length steps:1e5
```


