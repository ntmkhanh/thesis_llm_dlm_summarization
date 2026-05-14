import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


DEFAULT_LLM_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]
DEFAULT_LLM_NAMES = ["qwen", "llama"]


def run_cmd(cmd):
    print("\n$", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def parse_csv_arg(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run Method 1 Single-Draft with multiple fine-tuned draft LLMs "
            "followed by the same DiffuSeq/SeqDiffuSeq refiner."
        )
    )
    p.add_argument("--split", default="test")
    p.add_argument("--llm-models", default=",".join(DEFAULT_LLM_MODELS))
    p.add_argument("--llm-names", default=",".join(DEFAULT_LLM_NAMES))
    p.add_argument("--dlm-backbone", default="google/flan-t5-base")
    p.add_argument("--paper-mode", choices=["diffuseq", "seqdiffuseq"], default="diffuseq")
    p.add_argument("--diffusion-steps", type=int, default=8)
    p.add_argument("--train-models", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--exp-name", default="method1_draft_generators")
    p.add_argument("--out-dir", default="outputs/experiments")
    return p.parse_args()


def main():
    args = parse_args()
    llm_models = parse_csv_arg(args.llm_models)
    llm_names = parse_csv_arg(args.llm_names)
    if len(llm_models) != len(llm_names):
        raise ValueError("--llm-models and --llm-names must have the same number of items")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{args.exp_name}_{args.paper_mode}_{ts}"
    base_dir = Path(args.out_dir) / exp_id
    models_dir = base_dir / "models"
    drafts_dir = base_dir / "drafts"
    metrics_dir = base_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    drafts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dlm_dir = models_dir / f"dlm_refiner_{args.paper_mode}"
    timing = {}
    outputs = {}
    metrics = {}

    if args.train_models:
        timing["train_dlm_refiner_sec"] = run_cmd([
            "python3", "src/pipeline/train_dlm_refiner.py",
            "--model", args.dlm_backbone,
            "--paper-mode", args.paper_mode,
            "--diffusion-steps", str(args.diffusion_steps),
            "--output-dir", str(dlm_dir),
            "--max-train-samples", str(args.max_train_samples),
            "--max-val-samples", str(args.max_val_samples),
        ])

    for name, model in zip(llm_names, llm_models):
        llm_dir = models_dir / f"{name}_sft"
        out_csv = drafts_dir / f"method1_single_{name}_{args.paper_mode}.csv"
        metric_json = metrics_dir / f"method1_single_{name}_{args.paper_mode}.json"

        if args.train_models:
            timing[f"train_llm_{name}_sec"] = run_cmd([
                "python3", "src/pipeline/train_llm_sft.py",
                "--model", model,
                "--output-dir", str(llm_dir),
                "--max-train-samples", str(args.max_train_samples),
                "--max-val-samples", str(args.max_val_samples),
            ])

        timing[f"infer_method1_single_{name}_sec"] = run_cmd([
            "python3", "src/pipeline/infer_method1_llm_dlm.py",
            "--llm-model-dir", str(llm_dir),
            "--dlm-model-dir", str(dlm_dir),
            "--draft-mode", "single",
            "--paper-mode", args.paper_mode,
            "--diffusion-steps", str(args.diffusion_steps),
            "--split", args.split,
            "--max-samples", str(args.max_test_samples),
            "--output", str(out_csv),
        ])

        timing[f"eval_method1_single_{name}_sec"] = run_cmd([
            "python3", "src/evaluation/compute_metrics.py",
            "--input", str(out_csv),
            "--output-json", str(metric_json),
        ])
        outputs[f"method1_single_{name}_{args.paper_mode}"] = str(out_csv)
        metrics[f"method1_single_{name}_{args.paper_mode}"] = str(metric_json)

    manifest = {
        "exp_id": exp_id,
        "created_at": ts,
        "split": args.split,
        "llm_models": dict(zip(llm_names, llm_models)),
        "dlm_backbone": args.dlm_backbone,
        "paper_mode": args.paper_mode,
        "diffusion_steps": args.diffusion_steps,
        "train_models": args.train_models,
        "timing_seconds": timing,
        "outputs": outputs,
        "metrics": metrics,
    }
    with open(base_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nMethod 1 draft-generator study completed: {base_dir}")


if __name__ == "__main__":
    main()
