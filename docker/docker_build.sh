#!/bin/bash
username=$(id -u -n)
uid=$(id -u)
groupname=$(id -g -n)
gid=$(id -g)

docker build --build-arg USERNAME=${username} \
    --build-arg UID=${uid} \
    --build-arg GROUPNAME=${groupname} \
    --build-arg GID=${gid} \
    -t repo-luna.ist.osaka-u.ac.jp:5000/${username}/zeroshot-hierarchical:latest .