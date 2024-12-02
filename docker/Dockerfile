FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
MAINTAINER miyakawa

RUN useradd -u 1125 miyakawa

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install python packages
RUN apt-get update && apt-get install -y python3 python3-pip 

# install apt packages
RUN apt-get upgrade -y && apt-get install -y \
    build-essential \
    git \
    vim \
    sudo \
    wget \
    libgl1-mesa-dev \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxext-dev \
    libgl1-mesa-dev

# install requirements
RUN apt-get install git -y
# # install pytorch
# RUN python3 -m pip install --upgrade pip && python3 -m pip install \
#     torch==1.12.1+cu113 \
#     torchvision==0.13.1+cu113 \
#     torchaudio==0.12.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# for screen
RUN apt-get install screen -y

WORKDIR /home/miyakawa/mnt/workspace/zeroshot_hierarchical