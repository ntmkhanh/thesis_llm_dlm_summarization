import argparse
import os
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.load_cnn_dailymail import load_cnn
from src.llm.common import ensure_parent_dir, load_llm, summarize_article


def parse_args():
    parser = argparse.ArgumentParser(description="Generate baseline summaries with an LLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", default="test[:100]")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", default="outputs/drafts/baseline_qwen.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_cnn(splits=args.split, cnn_only=True)

    tokenizer, model = load_llm(args.model)

    rows = []
    for item in tqdm(dataset, desc="Generating summaries", total=len(dataset)):
        article = item["article"]
        reference = item["highlights"]
        summary = summarize_article(
            tokenizer,
            model,
            article,
            style="3-4 concise sentences",
            max_new_tokens=args.max_new_tokens,
        )
        rows.append({"article": article, "reference": reference, "summary": summary})

    ensure_parent_dir(args.output)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} summaries to {args.output}")


if __name__ == "__main__":
    main()
