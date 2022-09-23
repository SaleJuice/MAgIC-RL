# MAgIC-RL

A simple RL algorithms lib realized by pytorch.


## CMD
```
--job-type --train-env --eval-env --env-wrappers --render --buffer-size --agent --wb-dir --device --logger --project-name --group-name --verbose --experiment-length
```


## MAgIC-RL
### train
```
--job-type train --train-env Pendulum-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name MAgIC-RL --group-name Pendulum-v0 --experiment-length steps:1e5
```
### evaluate
```
--job-type eval --eval-env Pendulum-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/Pendulum-v0/steps_20200/ --device cuda:0 --project-name MAgIC-RL --group-name Pendulum-v0 --experiment-length episodes:3
```


## GymFish
### train
```
--job-type train --train-env gym_fish:T1-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name T1-v0 --experiment-length steps:1e5
```
### evaluate
```
--job-type eval --eval-env gym_fish:T1-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir checkpoints/1xme5own/steps_83587/ --device cuda:0 --project-name Gym-Fish --group-name T1-v0 --experiment-length episodes:3
```


## Try
```
--train-env GymFish-v4 --env-wrappers SequenceObservations:AddActionsToObservations:NormalizeActions --job-type train --agent sac_agent --buffer-size 1e6 --device cuda:0 --logger wandb --project-name Gym-Fish --group-name v4:so:aato:na --experiment-length steps:1e5
```
