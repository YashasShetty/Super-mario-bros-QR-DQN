import os
import datetime
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import QR_DQN
import torch.multiprocessing as _mp
import torch.nn.functional as F
import numpy as np
import shutil
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser("""Implementation of QR-DQN for training""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_local_steps', type=int, default=512)
    parser.add_argument('--num_global_steps', type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)         
    parser.add_argument('--save_interval', type=int, default=50, help="Number of episodes between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument('--log_path', type=str, default="tensorboard/qr_dqn_training")
    parser.add_argument('--saved_path', type=str, default="trained_models")
    args = parser.parse_args()
    return args

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    model = QR_DQN(envs.num_states, envs.envs[0].action_space.n, num_quantiles=32)  # Adjust num_quantiles as needed
    # print("envs.num_states",envs.num_states)
    # [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    # curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    # curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    curr_episode = 0
    episode_plot = []
    R_plot = []
    ep_reward_plot = []
    start_datetime = datetime.datetime.now().strftime("%m-%d_%H-%M")
    while curr_episode < opt.num_global_steps:
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model.state_dict(),
                       "{}/qr_dqn_trained_model_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        curr_episode += 1
        episode_plot.append(int(curr_episode))
        states = []
        actions = []
        rewards = []
        dones = []
        for _ in range(opt.num_local_steps):
            #dfjdf
            observations = [env.reset() for env in envs.envs]
            curr_states = torch.from_numpy(np.concatenate(observations, axis=0))

            # Reshape curr_states to match the expected input size of the first linear layer
            num_envs = len(envs.envs)
            num_features = observations[0].shape[0]  # Assuming all observations have the same shape
            batch_size = len(observations)  # This should be equal to num_envs if each environment contributes one observation
            curr_states = torch.tensor(observations, dtype=torch.float32).view(batch_size, num_features)

            print("curr_states.size",curr_states.size())
            print("num_envs",num_envs)
            curr_states = curr_states.view(num_envs, num_features)
            
            if torch.cuda.is_available():
                curr_states = curr_states.cuda()
            states.append(curr_states)
            quantiles = model(curr_states)
            q_values = torch.mean(quantiles, dim=2)
            actions.append(torch.argmax(q_values, dim=1))
            [agent_conn.send(("step", act.item())) for agent_conn, act in zip(envs.agent_conns, actions[-1])]
            results = [agent_conn.recv() for agent_conn in envs.agent_conns]
            state, reward, done, _ = zip(*results)
            rewards.append(torch.FloatTensor(reward))
            dones.append(torch.FloatTensor(done))

        next_states = torch.from_numpy(np.concatenate([env.reset() for env in envs], axis=0))
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        next_quantiles = model(next_states)
        next_q_values = torch.mean(next_quantiles, dim=2)
        next_actions = torch.argmax(next_q_values, dim=1)

        quantiles = model(torch.cat(states))
        actions_tensor = torch.cat(actions).squeeze()
        quantiles_actions = quantiles[torch.arange(actions_tensor.size(0)), actions_tensor]

        rewards_tensor = torch.cat(rewards).squeeze()
        dones_tensor = torch.cat(dones).squeeze()
        target_quantiles = rewards_tensor + opt.gamma * (1 - dones_tensor) * \
                           next_quantiles[torch.arange(next_actions.size(0)), next_actions]

        quantile_huber_loss = F.smooth_l1_loss(quantiles_actions, target_quantiles.detach(), reduction='none')
        k = 1.0 / opt.batch_size
        quantile_loss = torch.mean(torch.max((opt.tau - 1) * quantile_huber_loss, opt.tau * quantile_huber_loss), dim=1)
        loss = k * quantile_loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode: {curr_episode}, Loss: {loss.item()}")

    torch.save(model.state_dict(),
               "{}/qr_dqn_trained_model_{}_{}".format(opt.saved_path, opt.world, opt.stage))

if __name__ == "__main__":
    opt = get_args()
    train(opt)
