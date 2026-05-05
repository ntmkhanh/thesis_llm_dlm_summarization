import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class LatentDiffusionConfig:
    model_name_or_path: str = "google/flan-t5-base"
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    max_source_len: int = 1024
    max_target_len: int = 192
    self_condition: bool = True


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / max(1, half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class LatentDenoiser(nn.Module):
    """Predict noise in latent space.

    Inputs:
    - z_t: noisy summary latent [B, L, H]
    - article_ctx: pooled article context [B, H]
    - t: timestep [B]
    - self_cond: optional previous z0 prediction [B, L, H]
    """

    def __init__(self, hidden_size: int, self_condition: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_condition = self_condition
        in_size = hidden_size * (3 if self_condition else 2)
        self.t_embed = TimestepEmbedding(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
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

        ctx = article_ctx.unsqueeze(1).expand(b, l, h)
        t_emb = self.t_embed(t).unsqueeze(1).expand(b, l, h)
        x = torch.cat([feat, ctx + t_emb], dim=-1)
        return self.net(x)


class LatentDiffusionRefiner(nn.Module):
    """Latent diffusion refiner with frozen seq2seq backbone.

    This implements continuous latent diffusion over encoder hidden states:
    - q(z_t | z_0)
    - epsilon prediction in latent
    - reverse denoising
    - decode from denoised latent via seq2seq decoder
    """

    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path, device_map="auto")

        # freeze backbone to focus training on diffusion denoiser
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden = self.backbone.config.d_model
        self.denoiser = LatentDenoiser(hidden_size=hidden, self_condition=config.self_condition).to(self.backbone.device)

        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def device(self):
        return self.backbone.device

    def encode_text(self, texts, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.device)
        with torch.no_grad():
            enc = self.backbone.get_encoder()(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                return_dict=True,
            )
        return enc.last_hidden_state, tok["attention_mask"]

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # z_t = sqrt(alpha_bar_t) z_0 + sqrt(1-alpha_bar_t) eps
        a_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_eps(
        self,
        z_t: torch.Tensor,
        article_ctx: torch.Tensor,
        t: torch.Tensor,
        self_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.denoiser(z_t, article_ctx, t, self_cond=self_cond)

    def training_step(self, article_texts, summary_texts) -> torch.Tensor:
        z_article, _ = self.encode_text(article_texts, self.config.max_source_len)
        z0, _ = self.encode_text(summary_texts, self.config.max_target_len)

        b = z0.shape[0]
        t = torch.randint(0, self.config.timesteps, (b,), device=self.device)
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise)

        article_ctx = z_article.mean(dim=1)
        self_cond = None
        if self.config.self_condition and torch.rand(()) < 0.5:
            with torch.no_grad():
                eps_prev = self.predict_eps(zt, article_ctx, t, self_cond=None)
                a_bar = self.alpha_bars[t].view(-1, 1, 1)
                self_cond = (zt - torch.sqrt(1.0 - a_bar) * eps_prev) / torch.sqrt(a_bar)

        eps_hat = self.predict_eps(zt, article_ctx, t, self_cond=self_cond)
        return F.mse_loss(eps_hat, noise)

    def reverse_denoise(self, article_text: str, z_init: torch.Tensor) -> torch.Tensor:
        # z_init shape [1, L, H]
        z_article, _ = self.encode_text([article_text], self.config.max_source_len)
        article_ctx = z_article.mean(dim=1)

        zt = z_init
        self_cond = None
        for step in reversed(range(self.config.timesteps)):
            t = torch.tensor([step], device=self.device)
            eps_hat = self.predict_eps(zt, article_ctx, t, self_cond=self_cond)
            a_bar_t = self.alpha_bars[t].view(1, 1, 1)
            z0_hat = (zt - torch.sqrt(1.0 - a_bar_t) * eps_hat) / torch.sqrt(a_bar_t)

            if step > 0:
                a_bar_prev = self.alpha_bars[t - 1].view(1, 1, 1)
                zt = torch.sqrt(a_bar_prev) * z0_hat + torch.sqrt(1.0 - a_bar_prev) * eps_hat
            else:
                zt = z0_hat

            if self.config.self_condition:
                self_cond = z0_hat.detach()

        return zt

    def decode_from_latent(self, latent: torch.Tensor, max_new_tokens: int = 180) -> str:
        enc_out = BaseModelOutput(last_hidden_state=latent)
        with torch.no_grad():
            out = self.backbone.generate(
                encoder_outputs=enc_out,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        self.backbone.config.save_pretrained(output_dir)
        torch.save({
            "denoiser": self.denoiser.state_dict(),
            "config": self.config.__dict__,
        }, os.path.join(output_dir, "latent_diffusion.pt"))


def load_latent_diffusion_refiner(model_dir: str) -> LatentDiffusionRefiner:
    ckpt = torch.load(os.path.join(model_dir, "latent_diffusion.pt"), map_location="cpu")
    cfg = LatentDiffusionConfig(**ckpt["config"])
    cfg.model_name_or_path = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else cfg.model_name_or_path
    model = LatentDiffusionRefiner(cfg)
    model.denoiser.load_state_dict(ckpt["denoiser"])
    model.eval()
    return model
