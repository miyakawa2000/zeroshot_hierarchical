#!/bin/bash
username=$(id -u -n)
docker run --gpus all \
        -it -d \
        --shm-size=50G \
	-u $(id -u $username):$(id -g $username) \
        --name ${username}_zeroshot-hierarchical \
        -v /mnt/workspace2024/${username}:/home/${username}/mnt/workspace \
        -v ~/:/home/${username} \
        repo-luna.ist.osaka-u.ac.jp:5000/${username}/zeroshot-hierarchical:latest bash
