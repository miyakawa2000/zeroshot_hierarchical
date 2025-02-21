FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG USERNAME
ARG UID
ARG GROUPNAME
ARG GID
ARG USE_SHELL=/bin/bash
ARG PASSWORD=$USERNAME
ARG ROOT_PASSWORD='su'

MAINTAINER miyakawa

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y && apt upgrade -y
RUN apt install -y git vim sudo libgl1-mesa-dev

# set root password
RUN echo 'root:'$ROOT_PASSWORD | chpasswd

# make group and user
RUN groupadd -g $GID $GROUPNAME && \
    useradd -u $UID -g $GROUPNAME -G sudo -m -s $USE_SHELL $USERNAME && \
    echo $PASSWORD':'$PASSWORD | chpasswd $USERNAME

#install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext-dev \
    screen

# install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt