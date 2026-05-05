import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def run_cmd(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Run paper-aligned summarization study")
    p.add_argument("--split", default="test")
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dlm-model", default="google/flan-t5-base")
    p.add_argument("--paper-mode", choices=["diffuseq", "seqdiffuseq"], default="seqdiffuseq")
    p.add_argument("--diffusion-steps", type=int, default=8)
    p.add_argument("--train-models", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--exp-name", default="cnn_full_test")
    p.add_argument("--out-dir", default="outputs/experiments")
    return p.parse_args()


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{args.exp_name}_{ts}"

    base_dir = Path(args.out_dir) / exp_id
    drafts_dir = base_dir / "drafts"
    metrics_dir = base_dir / "metrics"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    llm_dir = str(models_dir / "llm_sft")
    dlm_dir = str(models_dir / "dlm_refiner")
    planner_dir = str(models_dir / "method2_planner")
    decoder_dir = str(models_dir / "method2_decoder")

    m1_csv = str(drafts_dir / "method1_llm_dlm.csv")
    m2_csv = str(drafts_dir / "method2_dlm_llm.csv")

    if args.train_models:
        run_cmd([
            "python3", "src/pipeline/train_llm_sft.py",
            "--model", args.llm_model,
            "--output-dir", llm_dir,
            "--max-train-samples", str(args.max_train_samples),
            "--max-val-samples", str(args.max_val_samples),
        ])

        run_cmd([
            "python3", "src/pipeline/train_dlm_refiner.py",
            "--model", args.dlm_model,
            "--paper-mode", args.paper_mode,
            "--diffusion-steps", str(args.diffusion_steps),
            "--output-dir", dlm_dir,
            "--max-train-samples", str(args.max_train_samples),
            "--max-val-samples", str(args.max_val_samples),
        ])

        run_cmd([
            "python3", "src/pipeline/train_method2_planner.py",
            "--model", args.dlm_model,
            "--output-dir", planner_dir,
            "--max-train-samples", str(args.max_train_samples),
            "--max-val-samples", str(args.max_val_samples),
        ])

        run_cmd([
            "python3", "src/pipeline/train_method2_decoder.py",
            "--model", args.dlm_model,
            "--output-dir", decoder_dir,
            "--max-train-samples", str(args.max_train_samples),
            "--max-val-samples", str(args.max_val_samples),
        ])

    run_cmd([
        "python3", "src/pipeline/infer_method1_llm_dlm.py",
        "--llm-model-dir", llm_dir,
        "--dlm-model-dir", dlm_dir,
        "--paper-mode", args.paper_mode,
        "--diffusion-steps", str(args.diffusion_steps),
        "--split", args.split,
        "--max-samples", str(args.max_test_samples),
        "--output", m1_csv,
    ])

    run_cmd([
        "python3", "src/pipeline/infer_method2_dlm_llm.py",
        "--planner-model-dir", planner_dir,
        "--decoder-model-dir", decoder_dir,
        "--split", args.split,
        "--max-samples", str(args.max_test_samples),
        "--output", m2_csv,
    ])

    for name, path in [
        ("method1_llm_dlm", m1_csv),
        ("method2_dlm_llm", m2_csv),
    ]:
        run_cmd([
            "python3", "src/evaluation/compute_metrics.py",
            "--input", path,
            "--output-json", str(metrics_dir / f"{name}.json"),
        ])

    manifest = {
        "exp_id": exp_id,
        "created_at": ts,
        "split": args.split,
        "llm_model": args.llm_model,
        "dlm_model": args.dlm_model,
        "paper_mode": args.paper_mode,
        "diffusion_steps": args.diffusion_steps,
        "train_models": args.train_models,
        "outputs": {
            "method1_llm_dlm": m1_csv,
            "method2_dlm_llm": m2_csv,
        },
        "metrics": {
            "method1_llm_dlm": str(metrics_dir / "method1_llm_dlm.json"),
            "method2_dlm_llm": str(metrics_dir / "method2_dlm_llm.json"),
        },
    }

    with open(base_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStudy completed: {base_dir}")


if __name__ == "__main__":
    main()
