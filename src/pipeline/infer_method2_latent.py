import argparse
import os
import re
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.genie_core import GenieDiffusionConfig, GenieGaussianDiffusion
from src.dlm.genie_denoiser import GenieDenoiser
from src.dlm.genie_grounding import decode_with_backbone
from src.pipeline.common import SPLIT_TEST, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Method2 pure latent diffusion inference")
    p.add_argument("--method2-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TEST)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--init-mode", choices=["pure_noise", "draft_latent"], default="pure_noise")
    p.add_argument("--t-small", type=int, default=12, help="Forward noise level for draft_latent init.")
    p.add_argument("--max-source-len", type=int, default=0, help="Override source length from checkpoint config (0=use ckpt).")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--output", default="outputs/drafts/method2_latent.csv")
    return p.parse_args()


def encode(backbone, tokenizer, texts, max_len, device):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc = backbone.get_encoder()(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], return_dict=True)
    return enc.last_hidden_state


def estimate_target_len(article: str, tokenizer, max_source_len: int, max_target_len: int) -> int:
    src_ids = tokenizer(
        [article],
        return_tensors="pt",
        truncation=True,
        max_length=max_source_len,
    )["input_ids"]
    src_len = int(src_ids.shape[1])
    est = max(24, int(src_len * 0.20))
    return min(max_target_len, est)


def build_lead_draft(article: str, max_chars: int = 600) -> str:
    text = " ".join(article.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    lead = " ".join(parts[:2]).strip()
    if not lead:
        lead = text[:max_chars]
    return lead[:max_chars]


def reverse_from_noise(article, x_t, backbone, tokenizer, denoiser, diffusion, cfg, device, start_step=None):
    with torch.inference_mode():
        h_s = encode(backbone, tokenizer, [article], cfg["max_source_len"], device)
        xt = x_t
        if start_step is None:
            start_step = cfg["timesteps"] - 1
        for step in reversed(range(start_step + 1)):
            t = torch.tensor([step], device=device)
            eps = denoiser(xt, t, h_s)
            xt = diffusion.p_step(xt, step, eps)
    return xt


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = load_cnn_split(args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    ckpt = torch.load(os.path.join(args.method2_model_dir, "method2_latent_denoiser.pt"), map_location="cpu")
    cfg = ckpt["config"]

    tok = AutoTokenizer.from_pretrained(args.method2_model_dir)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"], device_map="auto")
    device = backbone.device
    backbone.eval()

    denoiser = GenieDenoiser(cfg["hidden"], num_layers=cfg["denoiser_layers"], num_heads=cfg["denoiser_heads"]).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()

    diffusion = GenieGaussianDiffusion(GenieDiffusionConfig(timesteps=cfg["timesteps"]), device=device)

    rows = []
    for ex in tqdm(ds, total=len(ds), desc="Method2 latent"):
        article = ex["article"]
        reference = ex["highlights"]

        src_max_len = int(args.max_source_len) if args.max_source_len > 0 else int(cfg["max_source_len"])
        cfg_runtime = dict(cfg)
        cfg_runtime["max_source_len"] = src_max_len

        if args.init_mode == "draft_latent":
            draft = build_lead_draft(article)
            z0 = encode(backbone, tok, [draft], cfg["max_target_len"], device)
            start_step = max(0, min(cfg["timesteps"] - 1, int(args.t_small)))
            t = torch.tensor([start_step], device=device)
            noise = torch.randn_like(z0)
            x_t = diffusion.q_sample(z0, t, noise)
            x_hat = reverse_from_noise(
                article, x_t, backbone, tok, denoiser, diffusion, cfg_runtime, device, start_step=start_step
            )
        else:
            target_len = estimate_target_len(article, tok, src_max_len, cfg["max_target_len"])
            x_t = torch.randn((1, target_len, cfg["hidden"]), device=device)
            x_hat = reverse_from_noise(article, x_t, backbone, tok, denoiser, diffusion, cfg_runtime, device)

        with torch.inference_mode():
            summary = decode_with_backbone(backbone, tok, x_hat, max_new_tokens=args.max_new_tokens)

        rows.append({"article": article, "reference": reference, "summary": summary})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
