import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.core_diffusion import DiffusionSchedule
from src.dlm.latent_denoiser import LatentDenoiser
from src.dlm.latent_fusion_gating import load_gating
from src.pipeline.common import SPLIT_TEST, build_summarization_prompt, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Method1 latent diffusion inference: compare mean vs learned fusion")
    p.add_argument("--llm-model-dir", required=True)
    p.add_argument("--latent-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TEST)
    p.add_argument("--num-candidates", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--fusion", choices=["mean", "learned"], default="mean")
    p.add_argument("--gating-model", default="")
    p.add_argument("--output", default="outputs/drafts/method1_latent_mean.csv")
    return p.parse_args()


def llm_generate_candidates(tokenizer, model, article: str, n: int, max_new_tokens: int):
    prompt = build_summarization_prompt(article)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=max(1, n),
            pad_token_id=tokenizer.pad_token_id,
        )
    cands = []
    for i in range(out.shape[0]):
        txt = tokenizer.decode(out[i], skip_special_tokens=True)
        cands.append(txt.split("Summary:")[-1].strip())
    return cands


def encode(backbone, tokenizer, texts, max_len, device):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc = backbone.get_encoder()(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            return_dict=True,
        )
    return enc.last_hidden_state


def decode(backbone, tokenizer, latent, max_new_tokens):
    with torch.no_grad():
        out = backbone.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=latent),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def reverse_denoise(article, z_init, backbone, tokenizer, denoiser, schedule, cfg, device):
    z_article = encode(backbone, tokenizer, [article], cfg["max_source_len"], device)
    article_ctx = z_article.mean(dim=1)

    zt = z_init
    self_cond = None
    for step in reversed(range(cfg["timesteps"])):
        t = torch.tensor([step], device=device)
        eps_hat = denoiser(zt, article_ctx, t, self_cond=self_cond)
        z0_hat = schedule.predict_z0_from_eps(zt, t, eps_hat)
        zt = schedule.p_sample_step(zt, step, eps_hat)
        if cfg.get("self_condition", False):
            self_cond = z0_hat.detach()
    return zt


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = load_cnn_split(args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    llm_tok = AutoTokenizer.from_pretrained(args.llm_model_dir)
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model_dir, device_map="auto")

    ckpt = torch.load(os.path.join(args.latent_model_dir, "latent_denoiser.pt"), map_location="cpu")
    cfg = ckpt["config"]

    tok = AutoTokenizer.from_pretrained(args.latent_model_dir)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"], device_map="auto")
    device = backbone.device

    denoiser = LatentDenoiser(cfg["hidden"], self_condition=cfg.get("self_condition", False)).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()

    schedule = DiffusionSchedule(cfg["timesteps"], cfg["beta_start"], cfg["beta_end"], device=str(device))

    gating = None
    if args.fusion == "learned":
        if not args.gating_model:
            raise ValueError("--fusion learned requires --gating-model")
        gating = load_gating(args.gating_model, device=device).to(device)

    rows = []
    for ex in tqdm(ds, total=len(ds), desc=f"Method1 latent fusion={args.fusion}"):
        article = ex["article"]
        reference = ex["highlights"]

        drafts = llm_generate_candidates(llm_tok, llm, article, args.num_candidates, args.max_new_tokens)
        if not drafts:
            continue

        z_drafts = encode(backbone, tok, drafts, cfg["max_target_len"], device)  # [K,L,H]

        if args.fusion == "mean":
            z_fused = z_drafts.mean(dim=0, keepdim=True)
            weights = [1.0 / z_drafts.shape[0]] * z_drafts.shape[0]
        else:
            z_article = encode(backbone, tok, [article], cfg["max_source_len"], device)
            za = z_article.mean(dim=1).squeeze(0)
            zd_pool = z_drafts.mean(dim=1)
            with torch.no_grad():
                z_star, w = gating.fuse(za, zd_pool)
            sims = [F.cosine_similarity(zd_pool[i].unsqueeze(0), z_star.unsqueeze(0)).item() for i in range(zd_pool.shape[0])]
            best_idx = max(range(len(sims)), key=lambda i: sims[i])
            z_fused = z_drafts[best_idx : best_idx + 1]
            weights = w.detach().cpu().tolist()

        noise = torch.randn_like(z_fused)
        tmax = torch.tensor([cfg["timesteps"] - 1], device=device)
        z_t = schedule.q_sample(z_fused, tmax, noise)
        z_hat = reverse_denoise(article, z_t, backbone, tok, denoiser, schedule, cfg, device)
        summary = decode(backbone, tok, z_hat, args.max_new_tokens)

        rows.append({
            "article": article,
            "reference": reference,
            "drafts": " ||| ".join(drafts),
            "summary": summary,
            "fusion": args.fusion,
            "fusion_weights": str(weights),
        })

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
