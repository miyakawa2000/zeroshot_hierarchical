#!/bin/bash
username=$(id -u -n)
container_name="${username}_zeroshot-hierarchical"
docker run --gpus all \
        -it -d \
        --shm-size=50G \
	-u $(id -u $username):$(id -g $username) \
        --name "$container_name" \
        -v /mnt/workspace2024/${username}/zeroshot_hierarchical:/home/${username}/mnt/workspace/zeroshot_hierarchical \
        -v /mnt/workspace2024/${username}/dataset:/home/${username}/mnt/workspace/dataset \
        -v ~/:/home/${username} \
        repo-luna.ist.osaka-u.ac.jp:5000/${username}/zeroshot-hierarchical:latest bash
