import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = "finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen on the CNN-only subset of CNN/DailyMail for draft summary generation"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-train-samples", type=int, default=0, help="0 means use all CNN-only samples from the original train split")
    p.add_argument("--max-val-samples", type=int, default=0, help="0 means use all CNN-only samples from the original validation split")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--tuning-mode", choices=["full", "lora", "qlora"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable, "src/pipeline/train_llm_sft.py",
        "--model", args.model,
        "--output-dir", args.output_dir,
        "--split-mode", "native",
        "--split-seed", str(args.split_seed),
        "--max-train-samples", str(args.max_train_samples),
        "--max-val-samples", str(args.max_val_samples),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--save-total-limit", str(args.save_total_limit),
        "--max-length", str(args.max_length),
        "--tuning-mode", args.tuning_mode,
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")

    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
