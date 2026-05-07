import argparse
import os
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
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--output", default="outputs/drafts/method2_latent.csv")
    return p.parse_args()


def encode(backbone, tokenizer, texts, max_len, device):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc = backbone.get_encoder()(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], return_dict=True)
    return enc.last_hidden_state


def reverse_from_noise(article, x_t, backbone, tokenizer, denoiser, diffusion, cfg, device):
    h_s = encode(backbone, tokenizer, [article], cfg["max_source_len"], device)
    xt = x_t
    for step in reversed(range(cfg["timesteps"])):
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

    denoiser = GenieDenoiser(cfg["hidden"], num_layers=cfg["denoiser_layers"], num_heads=cfg["denoiser_heads"]).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()

    diffusion = GenieGaussianDiffusion(GenieDiffusionConfig(timesteps=cfg["timesteps"]), device=device)

    rows = []
    for ex in tqdm(ds, total=len(ds), desc="Method2 latent"):
        article = ex["article"]
        reference = ex["highlights"]

        # initialize from pure Gaussian noise in latent space
        # build a latent shape proxy from source length by encoding a short template
        template = ["summary"]
        z_template = encode(backbone, tok, template, cfg["max_target_len"], device)
        x_t = torch.randn_like(z_template)

        x_hat = reverse_from_noise(article, x_t, backbone, tok, denoiser, diffusion, cfg, device)
        summary = decode_with_backbone(backbone, tok, x_hat, max_new_tokens=args.max_new_tokens)

        rows.append({"article": article, "reference": reference, "summary": summary})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
