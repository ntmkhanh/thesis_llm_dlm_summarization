import argparse
import os
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.load_cnn_dailymail import load_cnn
from src.llm.common import (
    ensure_parent_dir,
    generate_text,
    load_llm,
    refine_summary_with_editor,
    select_best_candidate,
    summarize_article,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Method 1: LLM draft -> diffusion-like refinement")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", default="test[:100]")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--multi-mode", choices=["refine_each", "aggregate_then_refine"], default="refine_each")
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", default="outputs/drafts/method1_single.csv")
    return parser.parse_args()


def _generate_candidates(tokenizer, model, article: str, num_candidates: int, max_new_tokens: int):
    prompt = (
        "Generate a concise factual summary in 3-4 sentences.\\n\\n"
        f"Article: {article}\\n"
        "Summary:"
    )
    candidates = []
    for i in range(max(1, num_candidates)):
        text = generate_text(
            tokenizer,
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=max(1, min(num_candidates, 6)),
            do_sample=False,
        )
        summary = text.split("Summary:")[-1].strip()
        candidates.append(summary)
        prompt = (
            "Generate another summary variant with different phrasing but same key facts.\\n\\n"
            f"Article: {article}\\n"
            f"Previous summary: {summary}\\n"
            "New summary:"
        )
    return candidates


def main():
    args = parse_args()
    dataset = load_cnn(splits=args.split, cnn_only=True)
    tokenizer, model = load_llm(args.model)

    rows = []
    for item in tqdm(dataset, desc="Method1 generating", total=len(dataset)):
        article = item["article"]
        reference = item["highlights"]

        if args.mode == "single":
            draft = summarize_article(tokenizer, model, article, max_new_tokens=args.max_new_tokens)
            refined = refine_summary_with_editor(tokenizer, model, article, draft, max_new_tokens=args.max_new_tokens)
            summary = refined
            draft_pool = [draft]
        else:
            draft_pool = _generate_candidates(tokenizer, model, article, args.num_candidates, args.max_new_tokens)

            if args.multi_mode == "refine_each":
                refined_pool = [
                    refine_summary_with_editor(tokenizer, model, article, draft, max_new_tokens=args.max_new_tokens)
                    for draft in draft_pool
                ]
                summary = select_best_candidate(refined_pool, article)
            else:
                merged_draft = " ".join([f"({i+1}) {d}" for i, d in enumerate(draft_pool)])
                summary = refine_summary_with_editor(
                    tokenizer,
                    model,
                    article,
                    merged_draft,
                    max_new_tokens=args.max_new_tokens,
                )

        rows.append(
            {
                "article": article,
                "reference": reference,
                "summary": summary,
                "draft_count": len(draft_pool),
                "drafts": " ||| ".join(draft_pool),
            }
        )

    ensure_parent_dir(args.output)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} summaries to {args.output}")


if __name__ == "__main__":
    main()
