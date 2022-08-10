#!/bin/bash
# TODO add the outer loop to decrease nbr of queries
for q in 50 25 10
do
  for i in {1..3}
  do
     echo "queries: ${q}, run nbr ${i}"
     python train.py --algo ppo --env Walker2d-v3 --track --wandb-project-name PrefLearn --regularize --n_queries $q
     python train.py --algo ppo --env Walker2d-v3 --track --wandb-project-name PrefLearn --n_queries $q
  done
done

