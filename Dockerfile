FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update -y && apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg
RUN pip install gym-super-mario-bros==7.3.2 opencv-python==4.3.0.36 future==0.18.2 pyglet==1.5.7 matplotlib 
RUN pip install numpy==1.19.0  gym==0.23.1

RUN apt-get install sudo && \
    sudo apt update && sudo apt install -y git && \
    sudo apt update && sudo apt install -y nano && sudo apt install -y vim && \
    cd /home && \
#    git clone https://github.com/AnthonySong98/Super-Mario-Bros-PPO.git
    git clone https://github.com/YashasShetty/Playing_Super-Mario-Bros_using_reinforcement_learning.git

#WORKDIR /home/Super-mario-bros-PPO
