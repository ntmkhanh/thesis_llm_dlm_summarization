import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = "finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft"
DEFAULT_DRAFT_DIR = "outputs/drafts/qwen_cnn_dailymail"


def parse_args():
    p = argparse.ArgumentParser(description="Generate Qwen draft summaries for a CNN-only CNN/DailyMail split")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0, help="0 means use all CNN-only samples from the original split")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--output", default="")
    return p.parse_args()


def default_output(split: str) -> str:
    return f"{DEFAULT_DRAFT_DIR}/qwen_{split}_drafts.csv"


def main():
    args = parse_args()
    output = args.output or default_output(args.split)
    cmd = [
        "python3", "src/pipeline/generate_llm_drafts.py",
        "--llm-model-dir", args.model_dir,
        "--split", args.split,
        "--split-mode", "native",
        "--split-seed", str(args.split_seed),
        "--max-samples", str(args.max_samples),
        "--max-new-tokens", str(args.max_new_tokens),
        "--output", output,
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
