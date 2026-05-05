import argparse
import os
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.latent_fusion_gating import load_gating
from src.dlm.paper_aligned_diffusion import DiffusionConfig, PaperAlignedDLMRefiner
from src.pipeline.common import SPLIT_TEST, build_summarization_prompt, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Method1 inference: LLM draft -> DLM refine")
    p.add_argument("--llm-model-dir", required=True)
    p.add_argument("--dlm-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TEST)
    p.add_argument("--paper-mode", choices=["diffuseq", "seqdiffuseq"], default="seqdiffuseq")
    p.add_argument("--diffusion-steps", type=int, default=8)
    p.add_argument(
        "--draft-mode",
        choices=["single", "multi_refine_each", "multi_aggregate_latent"],
        default="single",
    )
    p.add_argument("--num-candidates", type=int, default=3)
    p.add_argument("--latent-fusion", choices=["mean", "learned"], default="mean")
    p.add_argument("--latent-fusion-model", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--output", default="outputs/drafts/method1_llm_dlm.csv")
    return p.parse_args()


def llm_generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Summary:")[-1].strip()


def llm_generate_candidates(tokenizer, model, article: str, n: int, max_new_tokens: int):
    """Generate n draft candidates using sampling for Multi-Draft methods."""
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


def overlap_f1(pred: str, article: str) -> float:
    p = pred.lower().split()
    a = article.lower().split()
    if not p or not a:
        return 0.0
    ac = {}
    for t in a:
        ac[t] = ac.get(t, 0) + 1
    m = 0
    for t in p:
        if ac.get(t, 0) > 0:
            m += 1
            ac[t] -= 1
    prec = m / len(p)
    rec = m / len(a)
    return 0.0 if prec + rec == 0 else (2 * prec * rec / (prec + rec))


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
        config=DiffusionConfig(
            mode=args.paper_mode,
            num_steps=args.diffusion_steps,
            max_new_tokens=args.max_new_tokens,
        ),
    )
    gating = None
    if args.latent_fusion == "learned":
        if not args.latent_fusion_model:
            raise ValueError("--latent-fusion learned requires --latent-fusion-model <path>")
        gating = load_gating(args.latent_fusion_model, device=dlm_refiner.model.device)
        gating = gating.to(dlm_refiner.model.device)

    rows = []
    for ex in tqdm(ds, total=len(ds), desc="Method1 LLM->DLM"):
        article = ex["article"]
        ref = ex["highlights"]
        if args.draft_mode == "single":
            # Proposal Figure 3: Single-Draft
            draft = llm_generate(llm_tok, llm, build_summarization_prompt(article), args.max_new_tokens)
            refined = dlm_refiner.refine(article, draft)
            rows.append({"article": article, "reference": ref, "draft": draft, "summary": refined, "draft_mode": "single"})
            continue

        drafts = llm_generate_candidates(llm_tok, llm, article, args.num_candidates, args.max_new_tokens)

        if args.draft_mode == "multi_refine_each":
            # Proposal Figure 4: Multi-Draft (i)
            refined_pool = [dlm_refiner.refine(article, d) for d in drafts]
            best = max(refined_pool, key=lambda s: overlap_f1(s, article))
            rows.append({
                "article": article,
                "reference": ref,
                "draft": drafts[0] if drafts else "",
                "drafts": " ||| ".join(drafts),
                "summary": best,
                "draft_mode": "multi_refine_each",
            })
        else:
            # Proposal Figure 5: Multi-Draft (ii), latent aggregation -> reverse denoise
            summary = dlm_refiner.aggregate_drafts_in_latent(
                article,
                drafts,
                fusion_mode=args.latent_fusion,
                gating=gating,
            )
            rows.append({
                "article": article,
                "reference": ref,
                "draft": drafts[0] if drafts else "",
                "drafts": " ||| ".join(drafts),
                "summary": summary,
                "draft_mode": "multi_aggregate_latent",
                "latent_fusion": args.latent_fusion,
            })

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
