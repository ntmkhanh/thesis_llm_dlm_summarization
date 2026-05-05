import argparse
import os
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.common import SPLIT_TEST, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Method2 inference: DLM planner -> decoder")
    p.add_argument("--planner-model-dir", required=True)
    p.add_argument("--decoder-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TEST)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--output", default="outputs/drafts/method2_dlm_llm.csv")
    return p.parse_args()


def generate(tok, model, text: str, max_new_tokens: int) -> str:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = load_cnn_split(args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    planner_tok = AutoTokenizer.from_pretrained(args.planner_model_dir)
    planner = AutoModelForSeq2SeqLM.from_pretrained(args.planner_model_dir, device_map="auto")

    decoder_tok = AutoTokenizer.from_pretrained(args.decoder_model_dir)
    decoder = AutoModelForSeq2SeqLM.from_pretrained(args.decoder_model_dir, device_map="auto")

    rows = []
    for ex in tqdm(ds, total=len(ds), desc="Method2 DLM->LLM"):
        article = ex["article"]
        reference = ex["highlights"]

        p_prompt = f"Create a 3-bullet global summary plan for this article.\\n\\nArticle: {article}\\n\\nPlan:"
        plan = generate(planner_tok, planner, p_prompt, max_new_tokens=128)

        d_prompt = (
            "Generate a factual 3-4 sentence summary from article and plan.\\n\\n"
            f"Article: {article}\\n\\nPlan:\\n{plan}\\n\\nSummary:"
        )
        summary = generate(decoder_tok, decoder, d_prompt, max_new_tokens=args.max_new_tokens)

        rows.append({"article": article, "reference": reference, "plan": plan, "summary": summary})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
