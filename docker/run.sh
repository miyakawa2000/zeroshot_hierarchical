#!/bin/bash
username=$(id -u -n)
title=$1
container_num=$2
container_name="${username}_${title}_${container_num}"
docker run --gpus all -it -d \
        --shm-size=16gb \
        --name "$container_name" \
	    -u $(id -u $username):$(id -g $username) \
        $title \
        bash
