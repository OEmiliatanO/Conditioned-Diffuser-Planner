import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import os
import argparse

from utilis import visualize_maze_and_trajectories, set_seed
from MAZE import UMAZE_ARR, MEDIUM_MAZE_ARR, LARGE_MAZE_ARR

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(inp_channels, out_channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.Mish()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.Mish()
        
        self.time_mlp = nn.Linear(embed_dim, out_channels)
        
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = out + self.time_mlp(t_emb).unsqueeze(-1)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        return out + self.residual_conv(x)

class TemporalUNet(nn.Module):
    def __init__(self, transition_dim, dim=32, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.transition_dim = transition_dim
        
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv1d(transition_dim, dim, kernel_size=5, padding=2)
        
        self.downs = nn.ModuleList([])
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, time_dim),
                ResidualTemporalBlock(dim_out, dim_out, time_dim),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if ind < len(in_out) - 1 else nn.Identity()
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim)
        
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, time_dim),
                ResidualTemporalBlock(dim_in, dim_in, time_dim),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if ind < len(in_out) - 1 else nn.Identity()
            ]))
        
        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x, time):
        """
        x: [batch_size, transition_dim, horizon]
        time: [batch_size]
        cond: [batch_size, cond_dim]
        """
        t = self.time_mlp(time)
        x = self.init_conv(x)
        h = []

        # Down
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        # Mid
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # Up
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # Skip connection
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)

        return self.final_conv(x)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)

class GaussianDiffusion1D(nn.Module):
    def __init__(self, model, seq_length, timesteps=1000):
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas) 
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def forward(self, x_start):
        b, t, d = x_start.shape
        device = x_start.device

        x_start = x_start.permute(0, 2, 1)

        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(b, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1)
        x_noisy = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

class GaussianDiffusion1D_with_Sampling(GaussianDiffusion1D):
    def __init__(self, model, seq_length, timesteps=1000, clip_denoised=True):
        super().__init__(model, seq_length, timesteps)
        self.clip_denoised = clip_denoised
        
        alphas = 1. - self.betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod - 1))
        
        posterior_variance = (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) * self.betas
        self.register_buffer('posterior_variance', posterior_variance)
        
        self.register_buffer('posterior_mean_coef1', (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas) * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x_t +
            self.posterior_mean_coef2[t].view(-1, 1, 1) * x_start
        )
        return posterior_mean

    @torch.no_grad()
    def p_sample(self, x, t):
        b = x.shape[0]
        noise_pred = self.model(x.permute(0, 2, 1), t).permute(0, 2, 1)
        
        x_recon = self.predict_start_from_noise(x, t, noise_pred)
        
        if self.clip_denoised:
            x_recon.clamp_(-3.0, 3.0) 
            
        model_mean = self.q_posterior(x_recon, x, t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_var_t = self.posterior_variance[t].view(-1, 1, 1)
            nonzero_mask = (t != 0).float().view(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def conditional_sample(self, shape, conditions, state_dim, guidance_scale=0.0):
        device = next(self.model.parameters()).device
        b, horizon, dim = shape
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)

            # === Step 1: Reconstruction Guidance ===
            if guidance_scale > 0 and i > 0:
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    noise_pred = self.model(
                        x_in.permute(0, 2, 1), t_tensor
                    ).permute(0, 2, 1)
                    
                    x_0_pred = (
                        self.sqrt_recip_alphas_cumprod[i] * x_in
                        - self.sqrt_recipm1_alphas_cumprod[i] * noise_pred
                    ).clamp(-3.0, 3.0)
                    
                    guidance_loss = torch.var(x_0_pred[:, :, :state_dim], dim=0).mean()
                    guidance_loss.backward()
                
                scale = self.sqrt_one_minus_alphas_cumprod[i] * guidance_scale
                x = (x - scale * x_in.grad).detach()

            # === Step 2: Standard DDPM reverse step ===
            x = self.p_sample(x, t_tensor)

            # === Step 3: Hard inpainting replacement ===
            t_prev = max(i - 1, 0)
            for t_step, clean_val in conditions.items():
                noise = torch.randn_like(clean_val)
                noisy_condition = (
                    self.sqrt_alphas_cumprod[t_prev] * clean_val.to(device)
                    + self.sqrt_one_minus_alphas_cumprod[t_prev] * noise
                )
                x[:, t_step, :state_dim] = noisy_condition

        return x.detach().cpu()
    
    def conditional_sample_with_intermediate(self, shape, conditions, state_dim, guidance_scale=0.0):
        device = next(self.model.parameters()).device
        b, horizon, dim = shape
        x = torch.randn(shape, device=device)
        intermediates = []

        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)

            if guidance_scale > 0 and i > 0:
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    noise_pred = self.model(
                        x_in.permute(0, 2, 1), t_tensor
                    ).permute(0, 2, 1)
                    
                    x_0_pred = (
                        self.sqrt_recip_alphas_cumprod[i] * x_in
                        - self.sqrt_recipm1_alphas_cumprod[i] * noise_pred
                    ).clamp(-3.0, 3.0)
                    
                    guidance_loss = torch.var(x_0_pred[:, :, :state_dim], dim=0).mean()
                    guidance_loss.backward()
                
                scale = self.sqrt_one_minus_alphas_cumprod[i] * guidance_scale
                x = (x - scale * x_in.grad).detach()

            x = self.p_sample(x, t_tensor)
            if i % 10 == 0:
                intermediates.append(x.detach().cpu())
            t_prev = max(i - 1, 0)
            for t_step, clean_val in conditions.items():
                noise = torch.randn_like(clean_val)
                noisy_condition = (
                    self.sqrt_alphas_cumprod[t_prev] * clean_val.to(device)
                    + self.sqrt_one_minus_alphas_cumprod[t_prev] * noise
                )
                x[:, t_step, :state_dim] = noisy_condition

        return x.detach().cpu(), intermediates

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
        0: normed_start_state.to(device),
        horizon - 1: normed_goal_state.to(device)
    }
    raw_conditions = {
        0: start_state.cpu(),
        horizon - 1: goal_state.cpu()
    }
    
    dangerous_zones = [
        # {'center': [6.0, 6.0], 'radius': 1.0},
        # {'center': [8.0, 4.8], 'radius': 1.15},
        # {'center': [7.0, 3.0], 'radius': 1.0},
    ]
    cost_fn   = make_cost_fn(dangerous_zones, mean.to(device), std.to(device), margin=0.5)
    reward_fn = make_reward_fn(mean.to(device), std.to(device))
    
    diffusion.eval()

    num_samples = args.num_samples
    sample_shape = (num_samples, horizon, transition_dim)
    cond = torch.cat([normed_start_state, normed_goal_state], dim=-1)
    cond = cond.expand(num_samples, -1).to(device)

    import time
    now = time.time()
    sampled_traj, intermediates = diffusion.conditional_sample_with_intermediate(
        shape=sample_shape,
        conditions=conditions,
        state_dim=pos_state_dim,
        guidance_scale=args.guidance_scale
    )
    print(f"Sampling took {time.time() - now:.2f} seconds for {num_samples} trajectories.")
    
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
    visualize_maze_and_trajectories(unnormed_sampled_traj.cpu(), MAZE_ARR, num_show=10, dangerous_zones=dangerous_zones, save_path=f"{args.save_dir}/{args.maze_size}_maze_all_sampled_trajectory.png")
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
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of trajectories to sample")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Scale for reconstruction guidance (default: 0.0, no guidance)")
    parser.add_argument("--save_raw_trajectories", action='store_true', help="Whether to save raw trajectory tensors for further analysis")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)