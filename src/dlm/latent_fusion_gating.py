import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LatentFusionConfig:
    hidden_dim: int
    max_candidates: int = 8


class LatentFusionGating(nn.Module):
    """Learnable gating for Multi-Draft latent fusion.

    Input per candidate i: [z_article ; z_draft_i ; z_article * z_draft_i ; |z_article-z_draft_i|]
    Output: unnormalized score s_i; weights = softmax(s)
    """

    def __init__(self, config: LatentFusionConfig):
        super().__init__()
        h = config.hidden_dim
        in_dim = h * 4
        self.config = config
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, 1),
        )

    def forward(self, z_article: torch.Tensor, z_drafts: torch.Tensor) -> torch.Tensor:
        """Return weights over candidates.

        z_article: [H]
        z_drafts: [K, H]
        """
        k, h = z_drafts.shape
        za = z_article.unsqueeze(0).expand(k, h)
        feats = torch.cat([za, z_drafts, za * z_drafts, (za - z_drafts).abs()], dim=-1)
        scores = self.scorer(feats).squeeze(-1)
        return torch.softmax(scores, dim=0)

    def fuse(self, z_article: torch.Tensor, z_drafts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.forward(z_article, z_drafts)  # [K]
        z_star = (w.unsqueeze(-1) * z_drafts).sum(dim=0)
        return z_star, w


def save_gating(model: LatentFusionGating, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "hidden_dim": model.config.hidden_dim,
            "max_candidates": model.config.max_candidates,
        },
    }, path)


def load_gating(path: str, device: Optional[str] = None) -> LatentFusionGating:
    ckpt = torch.load(path, map_location=device or "cpu")
    config = LatentFusionConfig(**ckpt["config"])
    model = LatentFusionGating(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def save_training_meta(path: str, meta: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
