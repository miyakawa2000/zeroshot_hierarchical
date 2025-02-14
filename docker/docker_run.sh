#!/bin/bash
username=$(id -u -n)
container_name="${username}_zeroshot-hierarchical"
docker run --gpus all -it -d \
        --shm-size=50G \
        --name "$container_name" \
	-u $(id -u $username):$(id -g $username) \
        -v /mnt/workspace2024/${username}/zeroshot_hierarchical:/home/${username}/mnt/workspace/zeroshot_hierarchical \
        repo-luna.ist.osaka-u.ac.jp:5000/${username}/zeroshot-hierarchical:latest bash
