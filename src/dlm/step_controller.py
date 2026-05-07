from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class StepControllerConfig:
    hidden_size: int
    step_bins: List[int]


class StepController(nn.Module):
    """Predict adaptive denoising budget T* from article/draft latent features."""

    def __init__(self, cfg: StepControllerConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.net = nn.Sequential(
            nn.Linear(h * 4, h),
            nn.ReLU(),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, len(cfg.step_bins)),
        )

    def forward(self, z_article: torch.Tensor, z_draft: torch.Tensor) -> torch.Tensor:
        # inputs: [B,H], [B,H]
        x = torch.cat([z_article, z_draft, z_article * z_draft, (z_article - z_draft).abs()], dim=-1)
        return self.net(x)

    def predict_steps(self, z_article: torch.Tensor, z_draft: torch.Tensor) -> torch.Tensor:
        logits = self.forward(z_article, z_draft)
        idx = logits.argmax(dim=-1)
        bins = torch.tensor(self.cfg.step_bins, device=logits.device)
        return bins[idx]


def save_step_controller(model: StepController, path: str):
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "hidden_size": model.cfg.hidden_size,
            "step_bins": model.cfg.step_bins,
        },
    }, path)


def load_step_controller(path: str, device: str = "cpu") -> StepController:
    ckpt = torch.load(path, map_location=device)
    cfg = StepControllerConfig(**ckpt["config"])
    model = StepController(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
