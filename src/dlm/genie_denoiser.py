import torch
import torch.nn as nn

from src.dlm.genie_core import sinusoidal_timestep_embedding


class GenieDenoiser(nn.Module):
    """GENIE-style denoiser: predict epsilon from (x_t, t, H_s).

    Uses TransformerDecoder blocks where x_t attends to source condition H_s.
    """

    def __init__(self, hidden_size: int, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.t_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        # x_t: [B, L_t, H], h_s: [B, L_s, H]
        t_emb = sinusoidal_timestep_embedding(t, self.hidden_size)
        t_emb = self.t_proj(t_emb).unsqueeze(1)  # [B,1,H]
        x = x_t + t_emb
        h = self.decoder(tgt=x, memory=h_s)
        return self.out(h)
