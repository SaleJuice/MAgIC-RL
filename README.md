# MAgIC-RL

A simple RL algorithms lib realized by pytorch.


## Arguments
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

## Swimmer-v3
### train
```
--job-type train --train-env Swimmer-v3 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Swimmer-v3 --group-name target_entorpy=-2 --experiment-length steps:1e6
```

### evaluate
```
--job-type eval --eval-env Swimmer-v3 --env-wrappers NormalizeActions --agent sac_agent --wb-dir checkpoints/2vrr80c2/steps_871000/ --device cuda:0 --render True --experiment-length episodes:1
```

## Hopper-v2
### train
```
--job-type train --train-env Hopper-v2 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Hopper-v2 --group-name debuged --experiment-length steps:1e6
```

### evaluate
```
--job-type eval --eval-env Hopper-v2 --env-wrappers NormalizeActions --agent sac_agent --wb-dir checkpoints/2vrr80c2/steps_871000/ --device cuda:0 --render True --experiment-length episodes:1
```


## GymFish
### train
```
--job-type train --train-env gym_fish:T1-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name T1-v0 --experiment-length steps:1e5
```
```
--job-type train --train-env gym_fish:T2-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name T2-v0 --experiment-length steps:1e5
```
### evaluate
```
--job-type eval --eval-env gym_fish:T1-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/T1-v0/steps_79171 --device cuda:0 --project-name Gym-Fish --group-name T1-v0 --experiment-length episodes:3
```
```
--job-type eval --eval-env gym_fish:T2-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir checkpoints/37y3t2qr/steps_34711/ --device cuda:0 --experiment-length episodes:3
```
--job-type train --train-env gym_fish:T8-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name IDEA8 --experiment-length steps:1e5

### Random Uniform Flow Adaption(RUFA) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:RUFA-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name RUFA-TRAIN --experiment-length steps:1e5
```
#### EVAL
```
--job-type eval --eval-env gym_fish:RUFA-v1 --env-wrappers NormalizeActions --device cuda:0 --experiment-length episodes:1
```
```
--job-type eval --eval-env gym_fish:RUFA-v1 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/rufa/random-energy/ --device cuda:0 --experiment-length episodes:1
```
#### RANDOM

### Avoid Obstacles in Flow Field(AOFF) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:AOFF-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name AOFF-Train-Static --experiment-length steps:1e5
```
#### EVAL
```
--job-type eval --eval-env gym_fish:AOFF-v1 --env-wrappers NormalizeActions --device cuda:0 --experiment-length episodes:1
```
```
--job-type eval --eval-env gym_fish:AOFF-v1 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/rufa/random-energy/ --device cuda:0 --experiment-length episodes:1
```
#### RANDOM

### Trajectory Tracking under Fluid Flow(TTFF) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:TTFF-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name TTFF-T-P-R --experiment-length steps:1e5
```
#### EVAL
```
--job-type eval --eval-env gym_fish:TTFF-v1 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/ttff/rpff/ --device cuda:0 --experiment-length episodes:1
```
```
--job-type eval --eval-env gym_fish:TTFF-v1 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/ttff/rprf/ --device cuda:0 --experiment-length episodes:1
```
#### RANDOM

### Pose Regulation Control Problem(PRCP) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:PRCP-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish --group-name PRCP-F --experiment-length steps:1e5
```
#### EVAL
```
--job-type eval --eval-env gym_fish:PRCP-v1 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/prcp/ --device cuda:0 --experiment-length episodes:1
```

### Point-to-point Navigation(PTPN) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:PTPN-Train-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish-New --group-name PTPN-Fixed-Target --experiment-length steps:1e5
```
#### EVAL
```
--job-type eval --eval-env gym_fish:PTPN-Eval-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/ptpn/steps_97497/ --device cuda:0 --experiment-length episodes:1
```


### Path-following Control(PFC) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:PFC-Train-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish-New --group-name PFC-Line --experiment-length steps:1e5
```
#### EVAL
```
--job-type eval --eval-env gym_fish:PFC-Eval-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/pfc/steps_22414/ --device cuda:0 --experiment-length episodes:1
```
yes
### Stay in the Circle(SITC) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:SITC-Train-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name Gym-Fish-New --group-name SITC-Random --experiment-length steps:1e5
```
#### EVAL
```
--job-type eval --eval-env gym_fish:SITC-Eval-v0 --env-wrappers NormalizeActions --device cuda:0 --experiment-length episodes:1
```
```
--job-type eval --eval-env gym_fish:SITC-Eval-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/sitc/random/steps_99658/ --device cuda:0 --experiment-length episodes:1
```

### Validation of Flow Field Information(VoFFI) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:SITC-Train-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name VoFFI --group-name baseline-without-speed --experiment-length steps:1e6
```
#### EVAL
```
--job-type eval --eval-env gym_fish:SITC-Eval-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/voffi/bws/steps_68038/ --device cuda:0 --experiment-length episodes:1
```

### Close With Target and Stay(CWTS) Experiment
#### TRAIN
```
--job-type train --train-env gym_fish:CWTS-Train-v0 --env-wrappers NormalizeActions --buffer-size 1e6 --agent sac_agent --device cuda:0 --logger wandb --project-name CWTS --group-name 30-noend-cwts --experiment-length steps:1e6
```
#### EVAL
```
--job-type eval --eval-env gym_fish:CWTS-Eval-v0 --env-wrappers NormalizeActions --agent sac_agent --wb-dir models/cwts/rew1/steps_104992/ --device cuda:0 --experiment-length episodes:1
```