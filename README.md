# Playing Super Mario Bros with Reinforcement Learning with QR DQN

Aim of this project is to improve the performance of Super Mario Bros playing agent. The baseline repo [link](https://github.com/AnthonySong98/Super-Mario-Bros-PPO) implements a the Quantile Regression Deep Q Network for creating a Mario playing agent. In attempts to improve the performance, we have added a layer to the neural network of the model.  

## How to run

* **Train your model** by running `python train.py`. For example: `python train.py --world 5 --stage 2 --lr 1e-4 --action_type complex`
* **Test your trained model** by running `python test.py`. For example: `python test.py --world 5 --stage 2 --action_type complex --iter 100`

For CSE server, do not forget add these two lines (tested).
```
Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:0
```
