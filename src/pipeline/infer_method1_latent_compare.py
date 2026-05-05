import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.latent_diffusion_refiner import load_latent_diffusion_refiner
from src.dlm.latent_fusion_gating import load_gating
from src.pipeline.common import SPLIT_TEST, build_summarization_prompt, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Compare latent mean vs learned fusion for Method1")
    p.add_argument("--llm-model-dir", required=True)
    p.add_argument("--ldm-model-dir", required=True, help="folder containing latent_diffusion.pt")
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

    ldm = load_latent_diffusion_refiner(args.ldm_model_dir)
    ldm = ldm.to(ldm.device)

    gating = None
    if args.fusion == "learned":
        if not args.gating_model:
            raise ValueError("--fusion learned requires --gating-model")
        gating = load_gating(args.gating_model, device=ldm.device).to(ldm.device)

    rows = []
    for ex in tqdm(ds, total=len(ds), desc=f"Method1 latent fusion={args.fusion}"):
        article = ex["article"]
        reference = ex["highlights"]

        drafts = llm_generate_candidates(llm_tok, llm, article, args.num_candidates, args.max_new_tokens)
        if not drafts:
            continue

        z_drafts = []
        for d in drafts:
            z, _ = ldm.encode_text([d], ldm.config.max_target_len)
            z_drafts.append(z.squeeze(0))
        z_drafts = torch.stack(z_drafts, dim=0)  # [K,L,H]

        if args.fusion == "mean":
            z_fused = z_drafts.mean(dim=0, keepdim=True)
            fusion_weights = [1.0 / z_drafts.shape[0]] * z_drafts.shape[0]
        else:
            z_article, _ = ldm.encode_text([article], ldm.config.max_source_len)
            za = z_article.mean(dim=1).squeeze(0)  # [H]
            zd_pool = z_drafts.mean(dim=1)  # [K,H]
            with torch.no_grad():
                z_star, w = gating.fuse(za, zd_pool)
            sims = [F.cosine_similarity(zd_pool[i].unsqueeze(0), z_star.unsqueeze(0)).item() for i in range(zd_pool.shape[0])]
            best_idx = max(range(len(sims)), key=lambda i: sims[i])
            z_fused = z_drafts[best_idx : best_idx + 1]
            fusion_weights = w.detach().cpu().tolist()

        # forward to z_T then reverse denoise
        noise = torch.randn_like(z_fused)
        tmax = torch.tensor([ldm.config.timesteps - 1], device=ldm.device)
        z_t = ldm.q_sample(z_fused, tmax, noise)
        z_hat = ldm.reverse_denoise(article, z_t)
        summary = ldm.decode_from_latent(z_hat, max_new_tokens=args.max_new_tokens)

        rows.append({
            "article": article,
            "reference": reference,
            "drafts": " ||| ".join(drafts),
            "summary": summary,
            "fusion": args.fusion,
            "fusion_weights": str(fusion_weights),
        })

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
