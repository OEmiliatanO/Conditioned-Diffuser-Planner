import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import os
import argparse

from diffusion import GaussianDiffusion1D_with_Sampling, TemporalUNet
from utilis import visualize_maze_and_trajectories
from MAZE import UMAZE_ARR, MEDIUM_MAZE_ARR, LARGE_MAZE_ARR

from utilis import set_seed

def evaluate_trajectories(trajectories, maze_arr, conditions, state_dim):
    valid_count = 0
    success_count = 0
    goal_state = conditions[max(conditions.keys())].cpu().numpy()
    start_state = conditions[0].cpu().numpy()
    valid_mask = torch.zeros(trajectories.shape[0], dtype=torch.bool)
    success_mask = torch.zeros(trajectories.shape[0], dtype=torch.bool)
    
    for i, traj in enumerate(trajectories):
        traj = traj.cpu().numpy()
        states = traj[:, :state_dim]
        
        # 檢查是否穿牆
        valid = True
        for state in states:
            x, y = state[0], state[1]
            r, c = int((maze_arr.shape[0] * 2 - y) // 2), int(x // 2)
            if 0 <= r < maze_arr.shape[0] and 0 <= c < maze_arr.shape[1]:
                if maze_arr[r, c] == 1:
                    valid = False
                    break
            else:
                valid = False
                break
        
        if valid:
            valid_count += 1
            valid_mask[i] = True
            
            if np.linalg.norm(states[1] - start_state) < 1 and np.linalg.norm(states[-2] - goal_state) < 1 \
                and np.linalg.norm(states[-1] - goal_state) < 1 and np.linalg.norm(states[0] - start_state) < 1:
                success_count += 1
                success_mask[i] = True

    valid_rate = valid_count / len(trajectories)
    success_rate = success_count / len(trajectories)
    diffs = torch.norm(
        trajectories[:, 1:, :state_dim] - 
        trajectories[:, :-1, :state_dim], 
        dim=-1
    )
    step_variance = diffs.var(dim=1).mean().item()
    return valid_rate, success_rate, step_variance, valid_mask, success_mask

def make_reward_fn(mean, std, state_dim=2):
    def reward_fn(x_0_pred):
        pos = x_0_pred[:, :, :state_dim]
        pos_real = pos * std[:state_dim] + mean[:state_dim]
        
        diffs = pos_real[:, 1:] - pos_real[:, :-1]        # [b, horizon-1, 2]
        length = -torch.norm(diffs, dim=-1).sum()
        length_std = torch.norm(diffs.detach(), dim=-1).sum(dim=-1).std()
        
        return length, length_std.item()
    
    return reward_fn

def make_cost_fn(dangerous_zones, mean, std, state_dim=2, margin=0.0):
    def cost_fn(x_0_pred):
        pos = x_0_pred[:, :, :state_dim]  # [b, horizon, 2]
        pos_real = pos * std[:state_dim] + mean[:state_dim]
        
        cost_per_sample = torch.zeros(pos_real.shape[0], device=pos_real.device)  # [b]
        total_cost = torch.zeros(1, device=x_0_pred.device)
        
        for zone in dangerous_zones:
            center = torch.tensor(zone['center'], device=x_0_pred.device)
            radius = zone['radius']
            
            dist = torch.norm(pos_real - center, dim=-1)  # [b, horizon]
            
            penetration = F.relu(radius + margin - dist)  # [b, horizon]
            
            cost_per_sample += penetration.detach().mean(dim=1)  # [b]
            total_cost = total_cost + penetration.mean()

        cost_std = cost_per_sample.std().item()

        return total_cost, cost_std
    
    return cost_fn

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pos_state_dim = 2
    vel_state_dim = 2
    state_dim = pos_state_dim + vel_state_dim
    action_dim = 2
    transition_dim = state_dim + action_dim
    
    horizon = 128

    unet = TemporalUNet(transition_dim=transition_dim).to(device)
    diffusion = GaussianDiffusion1D_with_Sampling(unet, seq_length=horizon, timesteps=1024).to(device)
    diffusion.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    
    data = torch.load(f"{args.maze_size}_maze_data.pth")
    MAZE_ARR = {
        "small": UMAZE_ARR,
        "medium": MEDIUM_MAZE_ARR,
        "large": LARGE_MAZE_ARR
    }[data["maze_size"]]
    mean = data['mean'].squeeze()
    std = data['std'].squeeze()
    
    diffusion.train()

    start_state = torch.tensor([3.0, 3.0])
    goal_state = torch.tensor([3.0, 7.0])
    normed_start_state = (start_state - mean[:pos_state_dim]) / std[:pos_state_dim]
    normed_goal_state = (goal_state - mean[:pos_state_dim]) / std[:pos_state_dim]

    conditions = {
        0: normed_start_state,
        horizon - 1: normed_goal_state
    }
    raw_conditions = {
        0: start_state.cpu(),
        horizon - 1: goal_state.cpu()
    }
    
    dangerous_zones = [
        {'center': [5.0, 7.0], 'radius': 0.5},
        {'center': [7.0, 5.0], 'radius': 0.5},
        {'center': [5.0, 3.0], 'radius': 0.3},
    ]
    cost_fn   = make_cost_fn(dangerous_zones, mean.to(device), std.to(device), margin=0.0)
    reward_fn = make_reward_fn(mean.to(device), std.to(device))
    
    diffusion.eval()

    num_samples = args.num_samples
    sample_shape = (num_samples, horizon, transition_dim)
    cond = torch.cat([normed_start_state, normed_goal_state], dim=-1)
    cond = cond.expand(num_samples, -1).to(device)

    import time
    now = time.time()
    sampled_traj, intermediates = diffusion.conditional_sample_with_intermediate(
        shape=(num_samples, horizon, transition_dim),
        cond=cond if args.conditioned else None,
        reward_fn=reward_fn,
        cost_fn=cost_fn,
        lambda_r=args.lambda_r,
        lambda_c=args.lambda_c,
        conflict_threshold=args.conflict_threshold,
        projection_enabled=args.projection_enabled
    )
    print(f"Sampling took {time.time() - now:.2f} seconds")
    
    unnormed_sampled_traj = sampled_traj * std + mean
    unnormed_intermediates = [intermediate * std + mean for intermediate in intermediates]
    
    valid_rate, success_rate, step_variance, valid_mask, success_mask = evaluate_trajectories(unnormed_sampled_traj.to(device), MAZE_ARR, raw_conditions, pos_state_dim)
    (valid_cost, valid_cost_std), (valid_reward, valid_reward_std) = cost_fn(sampled_traj[valid_mask].to(device)), reward_fn(sampled_traj[valid_mask].to(device))
    valid_cost = valid_cost.item()
    valid_reward = valid_reward.item()
    (success_cost, success_cost_std), (success_reward, success_reward_std) = cost_fn(sampled_traj[success_mask].to(device)), reward_fn(sampled_traj[success_mask].to(device))
    success_cost = success_cost.item()
    success_reward = success_reward.item()

    print(f"Valid Rate: {valid_rate:.4f} | Success Rate: {success_rate:.4f} | Step Variance: {step_variance:.4f}")
    print(f"Average Cost of Valid Trajectories: {valid_cost:.4f} ({valid_cost_std:.4f}) | Average Reward of Valid Trajectories: {valid_reward:.4f} ({valid_reward_std:.4f})")
    print(f"Average Cost of Successful Trajectories: {success_cost:.4f} ({success_cost_std:.4f}) | Average Reward of Successful Trajectories: {success_reward:.4f} ({success_reward_std:.4f})")

    os.makedirs(args.save_dir, exist_ok=True)
    visualize_maze_and_trajectories(unnormed_sampled_traj.cpu(), MAZE_ARR, num_show=min(args.num_samples, 10), dangerous_zones=dangerous_zones, save_path=f"{args.save_dir}/{args.maze_size}_maze_all_sampled_trajectory.png")
    os.makedirs(f"{args.save_dir}/samples", exist_ok=True)
    for i in range(0, len(unnormed_intermediates), 10):
        visualize_maze_and_trajectories(unnormed_intermediates[i].cpu(), MAZE_ARR, num_show=3, dangerous_zones=dangerous_zones, save_path=f"{args.save_dir}/samples/{args.maze_size}_maze_intermediate_trajectory_{i}.png", use_scatter=True)
    visualize_maze_and_trajectories(unnormed_intermediates[-1].cpu(), MAZE_ARR, num_show=3, dangerous_zones=dangerous_zones, save_path=f"{args.save_dir}/samples/{args.maze_size}_maze_intermediate_trajectory_{len(unnormed_intermediates)-1}.png", use_scatter=True)

    if args.save_raw_trajectories:
        np.save(f"{args.save_dir}/samples/{args.maze_size}_maze_raw_sampled_trajectories.npy", unnormed_sampled_traj.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze_size", type=str, default="small")
    parser.add_argument("--checkpoint_path", type=str, default="conditional_checkpoints/diffusion_final.pth")
    parser.add_argument("--save_dir", type=str, default="conditional_samples", help="Directory to save visualizations")
    parser.add_argument("--lambda_r", type=float, default=0.1, help="Reward guidance strength")
    parser.add_argument("--lambda_c", type=float, default=5.0, help="Cost guidance strength")
    parser.add_argument("--conflict_threshold", type=float, default=0.0, help="Cosine similarity threshold to detect reward-cost conflict")
    parser.add_argument("--projection_enabled", action='store_true', help="Whether to enable projection step to resolve reward-cost conflicts")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of trajectories to sample")
    parser.add_argument("--conditioned", action='store_true', help="Whether to condition on start and goal states")
    parser.add_argument("--save_raw_trajectories", action='store_true', help="Whether to save raw sampled trajectories as .npy files")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)