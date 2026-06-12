import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, TensorDataset
matplotlib.use('Agg')

from MAZE import UMAZE_ARR, MEDIUM_MAZE_ARR, LARGE_MAZE_ARR
from diffusion import GaussianDiffusion1D_with_Sampling, TemporalUNet
from utilis import visualize_maze_and_trajectories

def evaluate_trajectories(trajectories, MAZE_ARR, conditions, state_dim):
    valid_count = 0
    success_count = 0
    goal_state = conditions[max(conditions.keys())].cpu().numpy()
    start_state = conditions[0].cpu().numpy()
    
    for traj in trajectories:
        traj = traj.cpu().numpy()
        states = traj[:, :state_dim]
        
        valid = True
        for state in states:
            x, y = state[0], state[1]
            r, c = int((MAZE_ARR.shape[0] * 2 - y) // 2), int(x // 2) 
            if 0 <= r < MAZE_ARR.shape[0] and 0 <= c < MAZE_ARR.shape[1]:
                if MAZE_ARR[r, c] == 1:
                    valid = False
                    break
            else:
                valid = False
                break
        
        if valid:
            valid_count += 1
            
            if np.linalg.norm(states[1] - start_state) < 1 and np.linalg.norm(states[-2] - goal_state) < 1:
                success_count += 1

    valid_rate = valid_count / len(trajectories)
    success_rate = success_count / len(trajectories)
    return valid_rate, success_rate

def train(args):
    os.makedirs(f"{args.save_dir}/samples", exist_ok=True)
    os.makedirs(f"{args.save_dir}/checkpoints", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pos_state_dim = 2
    vel_state_dim = 2
    state_dim = pos_state_dim + vel_state_dim
    action_dim = 2
    transition_dim = state_dim + action_dim
    
    # data shape: [num_samples, horizon, transition_dim]
    data = torch.load(args.data_path)
    MAZE_ARR = {
        "small": UMAZE_ARR,
        "medium": MEDIUM_MAZE_ARR,
        "large": LARGE_MAZE_ARR
    }[data["maze_size"]]
    mean = data['mean'].squeeze()
    std = data['std'].squeeze()
    normalized_trajectories = (data['trajectories'] - mean) / std
    normalized_trajectories = torch.tensor(normalized_trajectories, dtype=torch.float32)
    dataset = TensorDataset(normalized_trajectories)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    horizon = normalized_trajectories.shape[1]
    unet = TemporalUNet(transition_dim=transition_dim).to(device)
    diffusion = GaussianDiffusion1D_with_Sampling(unet, seq_length=horizon, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=4e-5)
    diffusion.train()

    start_state = torch.tensor([3.0, 3.0])
    goal_state = torch.tensor([3.0, 7.0])
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    normed_start_state = (start_state - mean[:pos_state_dim]) / std[:pos_state_dim]
    normed_goal_state = (goal_state - mean[:pos_state_dim]) / std[:pos_state_dim]
    
    raw_conditions = {
        0: start_state.cpu(),
        horizon - 1: goal_state.cpu()
    }
    
    total_steps = args.total_steps
    progress_bar = tqdm.tqdm(range(total_steps), desc="Training Diffusion Model")
    current_step = 0
    while current_step < total_steps:
        epoch_loss = 0
        for batch in dataloader:
            trajectories = batch[0].to(device)
            
            optimizer.zero_grad()
            cond = torch.cat([
                trajectories[:, 0, :pos_state_dim],         # start
                trajectories[:, -1, :pos_state_dim]          # goal
            ], dim=-1)
            loss = diffusion(trajectories, cond=cond)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            current_step += 1
            progress_bar.update(1)

            if current_step % 1000 == 0:
                diffusion.eval()
                num_samples = 10
                sample_shape = (num_samples, horizon, transition_dim)
                cond = torch.cat([normed_start_state, normed_goal_state], dim=-1)
                cond = cond.expand(num_samples, -1).to(device)
                sampled_traj = diffusion.sample(sample_shape, cond=cond)
                
                unnormed_sampled_traj = sampled_traj * std + mean
                
                valid_rate, success_rate = evaluate_trajectories(unnormed_sampled_traj, MAZE_ARR, raw_conditions, pos_state_dim)
                print(f"step {current_step} | Valid Rate: {valid_rate:.1f} | Success Rate: {success_rate:.1f}")
                visualize_maze_and_trajectories(unnormed_sampled_traj.cpu(), MAZE_ARR, num_show=10, save_path=f"{args.save_dir}/samples/sampled_trajectory_step_{current_step}.png")

                torch.save(diffusion.state_dict(), f"{args.save_dir}/checkpoints/diffusion_step_{current_step}.pth")
                diffusion.train()
        
        print(f"step {current_step} | Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(diffusion.state_dict(), f"{args.save_dir}/checkpoints/diffusion_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model on U-Maze")
    parser.add_argument("--data_path", type=str, default="simulated_maze_data.npy", help="Path to the pre-generated dataset")
    parser.add_argument("--save_dir", type=str, default="conditional_checkpoints", help="Directory to save model checkpoints and visualizations")
    parser.add_argument("--timesteps", type=int, default=1024, help="Number of diffusion timesteps")
    parser.add_argument("--total_steps", type=int, default=10000, help="Total training steps")
    args = parser.parse_args()
    train(args)