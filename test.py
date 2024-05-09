import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import numpy as np
from src.env import create_train_env
from src.model import QR_DQN  # Import QR-DQN model
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: QR-DQN for Contra Nes""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--iter", type=str, default="")
    args = parser.parse_args()
    return args

def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    env = create_train_env(opt.world, opt.stage)  # Assuming the environment creation function doesn't require action input
    model = QR_DQN(env.observation_space.shape[0], env.action_space.n, num_quantiles=32)  # Initialize QR-DQN model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/qr_dqn_trained_model_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/qr_dqn_trained_model_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        quantiles = model(state)
        q_values = torch.mean(quantiles, dim=2)
        action = torch.argmax(q_values).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["flag_get"]:
            print("World {} stage {} completed".format(opt.world, opt.stage))
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)
