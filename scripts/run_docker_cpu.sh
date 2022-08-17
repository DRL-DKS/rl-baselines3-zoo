#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line
echo $pwd

docker run --ipc=host \
 --mount src=C:/Users/simon_hq0n8ve/Documents/projects/rl-baselines3-zoo,target=/root/code/rl_zoo,type=bind stablebaselines/rl-baselines3-zoo-cpu:latest\
  bash -c "cd /root/code/rl_zoo/ && $cmd_line"
