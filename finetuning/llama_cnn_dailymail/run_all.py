import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_MODEL_DIR = "finetuning/llama_cnn_dailymail/checkpoints/llama_cnn_dailymail_sft"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune LLaMA and generate CNN/DailyMail draft summaries")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--max-train-samples", type=int, default=0, help="0 means use all CNN-only samples from the original train split")
    p.add_argument("--max-val-samples", type=int, default=0, help="0 means use all CNN-only samples from the original validation split")
    p.add_argument("--max-test-samples", type=int, default=0, help="0 means use all CNN-only samples from the original test split")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--tuning-mode", choices=["full", "lora", "qlora"], default="lora")
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=80)
    return p.parse_args()


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    args = parse_args()
    if not args.skip_train:
        run([
            sys.executable, str(THIS_DIR / "train.py"),
            "--output-dir", args.model_dir,
            "--max-train-samples", str(args.max_train_samples),
            "--max-val-samples", str(args.max_val_samples),
            "--split-seed", str(args.split_seed),
            "--epochs", str(args.epochs),
            "--tuning-mode", args.tuning_mode,
            "--save-total-limit", str(args.save_total_limit),
        ])

    for split, max_samples in [
        ("train", args.max_train_samples),
        ("validation", args.max_val_samples),
        ("test", args.max_test_samples),
    ]:
        run([
            sys.executable, str(THIS_DIR / "generate_drafts.py"),
            "--model-dir", args.model_dir,
            "--split", split,
            "--max-samples", str(max_samples),
            "--split-seed", str(args.split_seed),
            "--max-new-tokens", str(args.max_new_tokens),
        ])


if __name__ == "__main__":
    main()
