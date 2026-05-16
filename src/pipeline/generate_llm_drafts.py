import argparse
import os
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.common import (
    SPLIT_MODE_CNN_70_20_10,
    SPLIT_MODE_NATIVE,
    SPLIT_TEST,
    build_summarization_prompt,
    load_cnn_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate draft summaries from a fine-tuned LLM")
    p.add_argument("--llm-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TEST)
    p.add_argument("--split-mode", choices=[SPLIT_MODE_NATIVE, SPLIT_MODE_CNN_70_20_10], default=SPLIT_MODE_NATIVE)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--output", required=True)
    return p.parse_args()


def generate_draft(tokenizer, model, article: str, max_new_tokens: int) -> str:
    prompt = build_summarization_prompt(article)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Summary:")[-1].strip()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = load_cnn_split(args.split, split_mode=args.split_mode, seed=args.split_seed)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_dir, device_map="auto")

    rows = []
    for ex in tqdm(ds, total=len(ds), desc=f"Generating LLM drafts ({args.split})"):
        article = ex["article"]
        rows.append({
            "article": article,
            "reference": ex["highlights"],
            "draft": generate_draft(tokenizer, model, article, args.max_new_tokens),
            "split": args.split,
            "split_mode": args.split_mode,
            "llm_model_dir": args.llm_model_dir,
        })

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} drafts to {args.output}")


if __name__ == "__main__":
    main()
