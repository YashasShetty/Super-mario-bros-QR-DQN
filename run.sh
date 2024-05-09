#!/bin/bash

set -e

xhost local:root

docker run --runtime=nvidia -it --rm \
    --privileged \
    --pid=host \
    -e SHELL\
    -e DISPLAY\
    -e DOCKER=1\
    --env="LOCAL_USER_ID=1000" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --gpus all ppo

#--env="DISPLAY=$DISPLAY" 
