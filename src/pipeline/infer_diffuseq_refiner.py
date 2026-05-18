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
    return p.parse_args()


def latest_sample_file(diffuseq_dir: str, model_dir: str) -> str:
    model_name = os.path.basename(os.path.normpath(model_dir))
    pattern = os.path.join(diffuseq_dir, "generation_outputs", model_name, "*.samples", "seed*_step*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No DiffuSeq sample files found with pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def clean_diffuseq_text(text: str) -> str:
    text = str(text or "")
    for token in ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "<pad>", "<s>", "</s>"):
        text = text.replace(token, " ")
    text = text.replace(" ##", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def samples_to_csv(sample_path: str, output: str):
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
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} decoded summaries to {output}")


def main():
    args = parse_args()
    diffuseq_dir = os.path.abspath(args.diffuseq_dir)
    model_dir = os.path.abspath(args.model_dir)

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
        samples_to_csv(sample_path, args.output)


if __name__ == "__main__":
    main()
