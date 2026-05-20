import argparse
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Decode summaries with a fine-tuned Shark-NLP/DiffuSeq refiner")
    p.add_argument("--diffuseq-dir", default="external/DiffuSeq")
    p.add_argument("--model-dir", required=True, help="DiffuSeq checkpoint folder under diffusion_models")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--step", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--pattern", default="ema")
    p.add_argument("--output", default="")
    p.add_argument("--expected-rows", type=int, default=0, help="Trim appended DiffuSeq samples to this row count. 0 infers from training_args.json.")
    p.add_argument("--skip-decode", action="store_true", help="Only convert the latest existing DiffuSeq sample file to CSV.")
    p.add_argument("--keep-samples", action="store_true", help="Do not delete old sample files before decoding.")
    return p.parse_args()


def latest_sample_file(diffuseq_dir: str, model_dir: str) -> str:
    model_name = os.path.basename(os.path.normpath(model_dir))
    pattern = os.path.join(diffuseq_dir, "generation_outputs", model_name, "*.samples", "seed*_step*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No DiffuSeq sample files found with pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def clear_existing_sample_files(diffuseq_dir: str, model_dir: str, seed: int):
    model_name = os.path.basename(os.path.normpath(model_dir))
    pattern = os.path.join(diffuseq_dir, "generation_outputs", model_name, "*.samples", f"seed{seed}_step*.json")
    files = glob.glob(pattern)
    for path in files:
        os.remove(path)
    if files:
        print(f"Removed {len(files)} old DiffuSeq sample file(s) before decoding")


def infer_expected_rows(model_dir: str, split: str) -> int:
    args_path = os.path.join(model_dir, "training_args.json")
    if not os.path.exists(args_path):
        return 0
    with open(args_path, "r", encoding="utf-8") as f:
        train_args = json.load(f)
    data_dir = train_args.get("data_dir", "")
    split_file = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"}[split]
    path = os.path.join(data_dir, split_file)
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def clean_diffuseq_text(text: str) -> str:
    text = str(text or "")
    for token in ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "<pad>", "<s>", "</s>"):
        text = text.replace(token, " ")
    text = text.replace(" ##", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def samples_to_csv(sample_path: str, output: str, expected_rows: int = 0):
    rows = []
    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            raw_recover = item.get("recover", "")
            rows.append({
                "summary": clean_diffuseq_text(raw_recover),
                "raw_summary": raw_recover,
                "reference": clean_diffuseq_text(item.get("reference", "")),
                "source": clean_diffuseq_text(item.get("source", "")),
            })
    if expected_rows > 0 and len(rows) > expected_rows:
        print(f"Sample file has {len(rows)} rows; keeping the last {expected_rows} rows to avoid appended duplicate decodes")
        rows = rows[-expected_rows:]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} decoded summaries to {output}")


def main():
    args = parse_args()
    diffuseq_dir = os.path.abspath(args.diffuseq_dir)
    model_dir = os.path.abspath(args.model_dir)

    if not args.skip_decode:
        if not args.keep_samples:
            clear_existing_sample_files(diffuseq_dir, model_dir, args.seed)
        cmd = [
            sys.executable, "-u", "scripts/run_decode.py",
            "--model_dir", model_dir,
            "--seed", str(args.seed),
            "--step", str(args.step),
            "--bsz", str(args.batch_size),
            "--split", args.split,
            "--pattern", args.pattern,
        ]
        print("$", " ".join(cmd))
        subprocess.run(cmd, cwd=diffuseq_dir, check=True)

    sample_path = latest_sample_file(diffuseq_dir, model_dir)
    print(f"Latest DiffuSeq sample file: {sample_path}")
    if args.output:
        expected_rows = args.expected_rows or infer_expected_rows(model_dir, args.split)
        samples_to_csv(sample_path, args.output, expected_rows=expected_rows)


if __name__ == "__main__":
    main()
