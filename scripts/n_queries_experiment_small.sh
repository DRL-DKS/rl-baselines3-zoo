#!/bin/bash
# TODO add the outer loop to decrease nbr of queries
for q in 25 10
do
  for i in {1..3}
  do
     echo "queries: ${q}, run nbr ${i}"
     python train.py --algo ppo --env HalfCheetah-v3 --track --wandb-project-name preflearn --regularize --n_queries $q
  done
done

