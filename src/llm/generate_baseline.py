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
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--output", default="outputs/drafts/baseline_qwen.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_cnn(splits=args.split, cnn_only=True)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    tokenizer, model = load_llm(args.model)

    rows = []
    for item in tqdm(dataset, desc="Generating summaries", total=len(dataset)):
        article = item["article"]
        reference = item["highlights"]
        summary = summarize_article(
            tokenizer,
            model,
            article,
            style=(
                "2-3 very short factual sentences (about 30-70 words total), "
                "only the most important facts, no extra details"
            ),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
        )
        rows.append({"article": article, "reference": reference, "summary": summary})

    ensure_parent_dir(args.output)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} summaries to {args.output}")


if __name__ == "__main__":
    main()
