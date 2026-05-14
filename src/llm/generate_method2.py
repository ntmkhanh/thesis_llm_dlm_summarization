import argparse
import os
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.load_cnn_dailymail import load_cnn
from src.llm.common import ensure_parent_dir, generate_text, load_llm


def parse_args():
    parser = argparse.ArgumentParser(description="Method 2: structure plan -> final decode")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--output", default="outputs/drafts/method2_plan_decode.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_cnn(splits=args.split, cnn_only=True)
    tokenizer, model = load_llm(args.model)

    rows = []
    for item in tqdm(dataset, desc="Method2 generating", total=len(dataset)):
        article = item["article"]
        reference = item["highlights"]

        plan_prompt = (
            "Create a global summary plan for the article as 3 bullet points: "
            "main event, critical details, and outcome/impact.\\n\\n"
            f"Article: {article}\\n\\n"
            "Plan:"
        )
        raw_plan = generate_text(
            tokenizer,
            model,
            plan_prompt,
            max_new_tokens=120,
            num_beams=1,
            do_sample=False,
        )
        plan = raw_plan.split("Plan:")[-1].strip()

        decode_prompt = (
            "Write a fluent 3-4 sentence summary from the following plan. "
            "Preserve factual consistency with the article.\\n\\n"
            f"Article: {article}\\n\\n"
            f"Plan: {plan}\\n\\n"
            "Summary:"
        )
        generated = generate_text(
            tokenizer,
            model,
            decode_prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        summary = generated.split("Summary:")[-1].strip()

        rows.append({"article": article, "reference": reference, "plan": plan, "summary": summary})

    ensure_parent_dir(args.output)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} summaries to {args.output}")


if __name__ == "__main__":
    main()
