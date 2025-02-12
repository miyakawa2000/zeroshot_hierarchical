FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
MAINTAINER miyakawa

RUN useradd -u 1125 miyakawa

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install python packages
RUN apt-get update && apt-get install -y python3 python3-pip 

#install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext-dev \
    libgl1-mesa-dev \
    screen

# install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /home/miyakawa/mnt/workspace/zeroshot_hierarchical