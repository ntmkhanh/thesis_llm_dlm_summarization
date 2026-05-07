import math
from dataclasses import dataclass

import torch


@dataclass
class GenieDiffusionConfig:
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02


class GenieGaussianDiffusion:
    """Continuous Gaussian diffusion core for GENIE-style latent denoising."""

    def __init__(self, cfg: GenieDiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1)
        return (xt - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)

    def p_step(self, xt: torch.Tensor, t_scalar: int, eps: torch.Tensor) -> torch.Tensor:
        t = torch.tensor([t_scalar], device=xt.device)
        x0_hat = self.x0_from_eps(xt, t, eps)
        if t_scalar > 0:
            a_prev = self.alpha_bars[t - 1].view(1, 1, 1)
            return torch.sqrt(a_prev) * x0_hat + torch.sqrt(1.0 - a_prev) * eps
        return x0_hat


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / max(1, half - 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb
