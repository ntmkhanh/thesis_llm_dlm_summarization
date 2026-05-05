import argparse
import os
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.latent_fusion_gating import (
    LatentFusionConfig,
    LatentFusionGating,
    save_gating,
    save_training_meta,
)
from src.dlm.paper_aligned_diffusion import DiffusionConfig, PaperAlignedDLMRefiner
from src.pipeline.common import SPLIT_TRAIN, build_summarization_prompt, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Train learnable latent fusion gating for Method1 Multi-Draft(ii)")
    p.add_argument("--llm-model-dir", required=True)
    p.add_argument("--dlm-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TRAIN)
    p.add_argument("--paper-mode", choices=["diffuseq", "seqdiffuseq"], default="seqdiffuseq")
    p.add_argument("--diffusion-steps", type=int, default=8)
    p.add_argument("--num-candidates", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="outputs/models/latent_fusion/gating.pt")
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


def overlap_f1(pred: str, ref: str) -> float:
    p = pred.lower().split()
    r = ref.lower().split()
    if not p or not r:
        return 0.0
    rc = {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    m = 0
    for t in p:
        if rc.get(t, 0) > 0:
            m += 1
            rc[t] -= 1
    prec = m / len(p)
    rec = m / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


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

    dlm_refiner = PaperAlignedDLMRefiner(
        model_name_or_path=args.dlm_model_dir,
        config=DiffusionConfig(mode=args.paper_mode, num_steps=args.diffusion_steps, max_new_tokens=args.max_new_tokens),
    )

    # Infer latent size from one encoding
    z0 = dlm_refiner.encode_text("latent probe")
    gate = LatentFusionGating(LatentFusionConfig(hidden_dim=int(z0.shape[0]), max_candidates=args.num_candidates)).to(z0.device)
    optim = torch.optim.Adam(gate.parameters(), lr=args.lr)

    gate.train()
    for ep in range(args.epochs):
        pbar = tqdm(ds, total=len(ds), desc=f"Train latent-gating ep{ep+1}")
        running = 0.0
        steps = 0
        for ex in pbar:
            article = ex["article"]
            ref = ex["highlights"]

            drafts = llm_generate_candidates(llm_tok, llm, article, args.num_candidates, args.max_new_tokens)
            if len(drafts) < 2:
                continue

            z_article = dlm_refiner.encode_text(article)
            z_drafts = torch.stack([dlm_refiner.encode_text(d) for d in drafts], dim=0)

            # pseudo target: candidate closer to reference gets higher weight
            targets = torch.tensor([overlap_f1(d, ref) for d in drafts], device=z_drafts.device)
            if torch.all(targets <= 0):
                continue
            target_w = torch.softmax(targets, dim=0)

            pred_w = gate(z_article, z_drafts)
            loss = torch.nn.functional.kl_div((pred_w + 1e-8).log(), target_w, reduction="batchmean")

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.item())
            steps += 1
            if steps % 10 == 0:
                pbar.set_postfix(loss=running / steps)

    gate.eval()
    save_gating(gate, args.output)
    save_training_meta(args.output + ".meta.json", {
        "split": args.split,
        "paper_mode": args.paper_mode,
        "diffusion_steps": args.diffusion_steps,
        "num_candidates": args.num_candidates,
        "epochs": args.epochs,
    })
    print(f"Saved latent fusion gating to {args.output}")


if __name__ == "__main__":
    main()
