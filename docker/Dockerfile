FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
MAINTAINER miyakawa

RUN useradd -u 1125 miyakawa

COPY requirements.txt .

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install python packages
RUN apt-get update && apt-get install -y python3 python3-pip 

# for cv2
RUN apt-get upgrade -y && apt-get install -y libgl1-mesa-dev libopencv-dev libglib2.0-0 libsm6 libxrender1 libxext6

# install requirements
RUN apt-get install git -y
RUN pip3 install -r requirements.txt

# for screen
RUN apt-get install screen -y

WORKDIR /home/miyakawa/mnt/workspace/zeroshot_hierarchical