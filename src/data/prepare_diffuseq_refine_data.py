import argparse
import json
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Convert LLM draft CSVs into Shark-NLP/DiffuSeq jsonl data. "
            "Each source is document + draft summary; target is reference summary."
        )
    )
    p.add_argument("--train-drafts", required=True)
    p.add_argument("--valid-drafts", required=True)
    p.add_argument("--test-drafts", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-article-words", type=int, default=360)
    p.add_argument("--max-draft-words", type=int, default=90)
    p.add_argument("--max-reference-words", type=int, default=90)
    return p.parse_args()


def truncate_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if max_words <= 0 or len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_source(article: str, draft: str, max_article_words: int, max_draft_words: int) -> str:
    article = truncate_words(article, max_article_words)
    draft = truncate_words(draft, max_draft_words)
    return f"document : {article} draft summary : {draft}"


def read_csvs(csv_paths: str) -> pd.DataFrame:
    paths = [p.strip() for p in csv_paths.split(",") if p.strip()]
    if not paths:
        raise ValueError("At least one CSV path is required")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def write_jsonl(
    csv_paths: str,
    out_path: str,
    max_article_words: int,
    max_draft_words: int,
    max_reference_words: int,
) -> int:
    df = read_csvs(csv_paths)
    required = {"article", "draft", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_paths} is missing required columns: {sorted(missing)}")

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            src = build_source(row["article"], row["draft"], max_article_words, max_draft_words)
            trg = truncate_words(row["reference"], max_reference_words)
            print(json.dumps({"src": src, "trg": trg}, ensure_ascii=False), file=f)
            n += 1
    return n


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    counts = {
        "train": write_jsonl(
            args.train_drafts,
            os.path.join(args.output_dir, "train.jsonl"),
            args.max_article_words,
            args.max_draft_words,
            args.max_reference_words,
        ),
        "valid": write_jsonl(
            args.valid_drafts,
            os.path.join(args.output_dir, "valid.jsonl"),
            args.max_article_words,
            args.max_draft_words,
            args.max_reference_words,
        ),
        "test": write_jsonl(
            args.test_drafts,
            os.path.join(args.output_dir, "test.jsonl"),
            args.max_article_words,
            args.max_draft_words,
            args.max_reference_words,
        ),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"counts": counts}, f, indent=2, ensure_ascii=False)
    print(f"Saved DiffuSeq refine data to {args.output_dir}: {counts}")


if __name__ == "__main__":
    main()
