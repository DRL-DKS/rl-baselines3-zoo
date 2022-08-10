#!/bin/bash
# TODO add the outer loop to decrease nbr of queries
for t in 60 70 80
do
  for i in {1..2}
  do
     echo "Truth: ${t}, run nbr ${i}"
     python train.py --algo ppo --env HalfCheetah-v3 --track --wandb-project-name PrefLearn --regularize --truth $t --n_queries 50
     python train.py --algo ppo --env HalfCheetah-v3 --track --wandb-project-name PrefLearn --truth $t --n_queries 50
  done
done

