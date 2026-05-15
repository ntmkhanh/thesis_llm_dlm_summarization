import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.pipeline.common import load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Verify that only CNN articles are loaded from CNN/DailyMail")
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--show", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_cnn_split(args.split)
    bad = [i for i, ex in enumerate(ds) if not (ex["article"] or "").lstrip().startswith("(CNN)")]
    print(f"split={args.split}")
    print(f"cnn_only_samples={len(ds)}")
    print(f"non_cnn_samples_after_filter={len(bad)}")
    for i in range(min(args.show, len(ds))):
        print(f"\n--- sample {i} ---")
        print(ds[i]["article"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
