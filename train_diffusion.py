import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

from MAZE import UMAZE_ARR, MEDIUM_MAZE_ARR, LARGE_MAZE_ARR
from utilis import visualize_maze_and_trajectories

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
        self.clip_denoised = clip_denoised # 開啟 x_0 截斷保護機制
        
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
                    
                    guidance_loss = sum(
                        F.mse_loss(
                            x_0_pred[:, t_step, :state_dim],
                            clean_val.to(device).unsqueeze(0).expand(b, -1)
                        )
                        for t_step, clean_val in conditions.items()
                    )
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

def evaluate_trajectories(trajectories, maze_arr, conditions, state_dim):
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
    
    horizon = 128

    unet = TemporalUNet(transition_dim=transition_dim).to(device)
    diffusion = GaussianDiffusion1D_with_Sampling(unet, seq_length=horizon, timesteps=args.timesteps).to(device)
    
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=4e-5)
    
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
    
    epochs = 1000
    total_steps = args.total_steps
    diffusion.train()

    start_state = torch.tensor([3.0, 21.0])
    goal_state = torch.tensor([15.0, 3.0])
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
    
    progress_bar = tqdm.tqdm(range(total_steps), desc="Training Diffusion Model")
    current_step = 0
    while current_step < total_steps:
        epoch_loss = 0
        for batch in dataloader:
            trajectories = batch[0].to(device)
            
            optimizer.zero_grad()
            loss = diffusion(trajectories)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            current_step += 1
            progress_bar.update(1)

            if current_step % 1000 == 0:
                diffusion.eval()
                sample_shape = (10, horizon, transition_dim)
                sampled_traj = diffusion.conditional_sample(sample_shape, conditions, pos_state_dim)
                
                unnormed_sampled_traj = sampled_traj * std + mean
                
                valid_rate, success_rate = evaluate_trajectories(unnormed_sampled_traj, MAZE_ARR, raw_conditions, pos_state_dim)
                print(f"step {current_step} | Valid Rate: {valid_rate:.4f} | Success Rate: {success_rate:.4f}")
                visualize_maze_and_trajectories(unnormed_sampled_traj.cpu(), MAZE_ARR, num_show=10, save_path=f"{args.save_dir}/samples/sampled_trajectory_step_{current_step}.png")

                torch.save(diffusion.state_dict(), f"{args.save_dir}/checkpoints/diffusion_step_{current_step}.pth")
                diffusion.train()
        
        print(f"step {current_step} | Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(diffusion.state_dict(), f"{args.save_dir}/checkpoints/diffusion_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model on U-Maze")
    parser.add_argument("--data_path", type=str, default="simulated_maze_data.npy", help="Path to the pre-generated dataset")
    parser.add_argument("--save_dir", type=str, default="unconditional_checkpoints", help="Directory to save model checkpoints and visualizations")
    parser.add_argument("--timesteps", type=int, default=1024, help="Number of diffusion timesteps")
    parser.add_argument("--total_steps", type=int, default=10000, help="Total training steps")
    args = parser.parse_args()
    train(args)