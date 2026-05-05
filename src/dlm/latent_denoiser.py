import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / max(1, half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class LatentDenoiser(nn.Module):
    """Timestep-conditioned denoiser epsilon_theta(z_t, t, cond)."""

    def __init__(self, hidden_size: int, self_condition: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_condition = self_condition

        feat_dim = hidden_size * (3 if self_condition else 2)
        self.t_embed = TimestepEmbedding(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        article_ctx: torch.Tensor,
        t: torch.Tensor,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, h = z_t.shape
        if self.self_condition:
            if self_cond is None:
                self_cond = torch.zeros_like(z_t)
            feat = torch.cat([z_t, self_cond, z_t - self_cond], dim=-1)
        else:
            feat = torch.cat([z_t, z_t], dim=-1)

        cond = article_ctx.unsqueeze(1).expand(b, l, h)
        t_emb = self.t_embed(t).unsqueeze(1).expand(b, l, h)
        return self.net(torch.cat([feat, cond + t_emb], dim=-1))
