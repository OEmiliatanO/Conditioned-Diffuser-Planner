import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        # 第一層卷積區塊
        self.conv1 = nn.Conv1d(inp_channels, out_channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.Mish()
        
        # 第二層卷積區塊
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.Mish()
        
        # 時間嵌入映射
        self.time_mlp = nn.Linear(embed_dim, out_channels)
        
        # 殘差連接維度對齊
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        # 根據論文，時間嵌入加在第一層卷積的激活中
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = out + self.time_mlp(t_emb).unsqueeze(-1)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        return out + self.residual_conv(x)

class TemporalUNet(nn.Module):
    def __init__(self, transition_dim, cond_dim=4, dim=32, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.transition_dim = transition_dim
        
        # 時間嵌入
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )

        # 初始卷積
        self.init_conv = nn.Conv1d(transition_dim, dim, kernel_size=5, padding=2)
        
        # Downsample 階段
        self.downs = nn.ModuleList([])
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, time_dim),
                ResidualTemporalBlock(dim_out, dim_out, time_dim),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if ind < len(in_out) - 1 else nn.Identity()
            ]))
            
        # Middle 階段
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim)
        
        # Upsample 階段
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, time_dim),
                ResidualTemporalBlock(dim_in, dim_in, time_dim),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if ind < len(in_out) - 1 else nn.Identity()
            ]))
            
        # 最終輸出映射回原本的維度 (state_dim + action_dim)
        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x, time, cond=None):
        """
        x: [batch_size, transition_dim, horizon]
        time: [batch_size]
        cond: [batch_size, cond_dim]
        """
        t = (self.time_mlp(time) + self.cond_mlp(cond)) if cond is not None else self.time_mlp(time)
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
    """
    Nichol & Dhariwal 所提出的 Cosine 排程
    """
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

    def forward(self, x_start, cond=None):
        """
        訓練時的前向過程與 Loss 計算
        x_start: 乾淨的軌跡數據 [batch_size, horizon, transition_dim]
        cond: [batch_size, cond_dim]
        """
        b, t, d = x_start.shape
        device = x_start.device

        x_start = x_start.permute(0, 2, 1)
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(b, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1)
        x_noisy = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        predicted_noise = self.model(x_noisy, t, cond)

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
        
        # 用於從 x_t 預測 x_0 的係數
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod - 1))
        
        # 用於計算後驗均值 (posterior mean) 的係數
        posterior_variance = (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) * self.betas
        self.register_buffer('posterior_variance', posterior_variance)
        
        self.register_buffer('posterior_mean_coef1', (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas) * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        """利用網路預測的雜訊，反推真實的 x_0"""
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """基於截斷後的 x_0 與目前的 x_t，計算最穩定的後驗均值"""
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x_t +
            self.posterior_mean_coef2[t].view(-1, 1, 1) * x_start
        )
        return posterior_mean

    @torch.no_grad()
    def p_sample(self, x, t, cond=None):
        b = x.shape[0]
        noise_pred = self.model(x.permute(0, 2, 1), t, cond).permute(0, 2, 1)
        
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
    def p_sample_with_guidance(self, x, t, cond, reward_fn=None, cost_fn=None,
                                lambda_r=1.0, lambda_c=5.0, conflict_threshold=0.0, projection_enabled=True):
        b = x.shape[0]
        i = t[0].item()

        # Step 1: 預測 noise 和 x_0
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            noise_pred = self.model(x_in.permute(0,2,1), t, cond).permute(0,2,1)
            
            x_recon = (
                self.sqrt_recip_alphas_cumprod[i] * x_in
                - self.sqrt_recipm1_alphas_cumprod[i] * noise_pred
            ).clamp(-3.0, 3.0)

            # Step 2: 在 x_recon 上計算 guidance 梯度
            g_R = torch.zeros_like(x_in)
            g_C = torch.zeros_like(x_in)

            if cost_fn is not None:
                cost_val, _ = cost_fn(x_recon)
                if cost_val.requires_grad:
                    cost_val.backward(retain_graph=True)
                    g_C = x_in.grad.clone()
                    x_in.grad.zero_()

            if reward_fn is not None:
                reward_val, _ = reward_fn(x_recon)
                reward_val = -reward_val
                reward_val.backward()
                g_R = x_in.grad.clone() if x_in.grad is not None else g_R

        # Step 3: 投影解衝突
        if reward_fn is not None and cost_fn is not None and projection_enabled:
            g_R_flat = g_R.reshape(b, -1)
            g_C_flat = g_C.reshape(b, -1)
            dot_RC    = (g_R_flat * g_C_flat).sum(dim=-1)
            norm_C_sq = (g_C_flat ** 2).sum(dim=-1) + 1e-8
            cos_sim   = dot_RC / (g_R_flat.norm(dim=-1) * g_C_flat.norm(dim=-1) + 1e-8)
            conflict_mask = (cos_sim < conflict_threshold).float().view(b, 1)
            # print(f"Timestep {i}: avg cos_sim={cos_sim.mean().item():.4f}, conflict_ratio={conflict_mask.mean().item():.4f}")
            proj_coef = (dot_RC / norm_C_sq).view(b, 1)
            g_R = (g_R_flat - conflict_mask * proj_coef * g_C_flat).reshape_as(g_R)

        # Step 4: 直接修正 x，讓修正後的 x 進入 posterior 計算
        # 注意：這裡不用 sqrt_one_minus_alphas_cumprod 縮放
        # 改用固定的 step_size，讓各 timestep 的效果一致
        x_guided = x - (lambda_r * g_R + lambda_c * g_C)

        # Step 5: 用修正後的 x 重新算 x_recon 和 posterior mean
        with torch.no_grad():
            noise_pred2 = self.model(x_guided.permute(0,2,1), t, cond).permute(0,2,1)
            # x_recon2 = (
            #     self.sqrt_recip_alphas_cumprod[i] * x_guided
            #     - self.sqrt_recipm1_alphas_cumprod[i] * noise_pred2
            # ).clamp(-3.0, 3.0)
            x_recon2 = self.predict_start_from_noise(x_guided, t, noise_pred2).clamp(-3.0, 3.0)
            model_mean = self.q_posterior(x_recon2, x_guided, t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_var_t = self.posterior_variance[t].view(-1, 1, 1)
            nonzero_mask = (t != 0).float().view(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, shape, cond=None):
        device = next(self.model.parameters()).device
        b, horizon, dim = shape
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, cond)

        return x.detach().cpu()
    
    @torch.no_grad()
    def conditional_sample(
        self, shape, cond,
        reward_fn,       # callable: x_0_pred → scalar reward
        cost_fn,         # callable: x_0_pred → scalar cost
        lambda_r=1.0,
        lambda_c=5.0,
        conflict_threshold=0.0,  # cos < 0 就投影
    ):
        device = next(self.model.parameters()).device
        b, horizon, dim = shape
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)

            x = self.p_sample_with_guidance(x, t_tensor, cond, reward_fn=reward_fn, cost_fn=cost_fn,
                                            lambda_r=lambda_r, lambda_c=lambda_c, conflict_threshold=conflict_threshold)

        return x.detach().cpu()
    
    @torch.no_grad()
    def sample_with_intermediate(self, shape, cond=None):
        device = next(self.model.parameters()).device
        b, horizon, dim = shape
        x = torch.randn(shape, device=device)
        intermediates = []

        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, cond)

            if i % 10 == 0:
                intermediates.append(x.detach().cpu())

        return x.detach().cpu(), intermediates
    
    @torch.no_grad()
    def conditional_sample_with_intermediate(self, shape, cond,
        reward_fn, cost_fn, lambda_r=1.0, lambda_c=5.0, conflict_threshold=0.0, projection_enabled=True):
        device = next(self.model.parameters()).device
        b, horizon, dim = shape
        x = torch.randn(shape, device=device)
        intermediates = []

        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample_with_guidance(x, t_tensor, cond, reward_fn=reward_fn, cost_fn=cost_fn,
                                            lambda_r=lambda_r, lambda_c=lambda_c, conflict_threshold=conflict_threshold, projection_enabled=projection_enabled)

            if i % 10 == 0:
                intermediates.append(x.detach().cpu())

        return x.detach().cpu(), intermediates