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
    p = argparse.ArgumentParser(description="Run a structured summarization study")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
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

    baseline_csv = str(drafts_dir / "baseline.csv")
    m1_csv = str(drafts_dir / "method1_single.csv")
    m2_csv = str(drafts_dir / "method2_plan_decode.csv")

    run_cmd([
        "python3", "src/llm/generate_baseline.py",
        "--model", args.model,
        "--split", args.split,
        "--output", baseline_csv,
    ])

    run_cmd([
        "python3", "src/llm/generate_method1.py",
        "--model", args.model,
        "--mode", "single",
        "--split", args.split,
        "--output", m1_csv,
    ])

    run_cmd([
        "python3", "src/llm/generate_method2.py",
        "--model", args.model,
        "--split", args.split,
        "--output", m2_csv,
    ])

    for name, path in [
        ("baseline", baseline_csv),
        ("method1_single", m1_csv),
        ("method2_plan_decode", m2_csv),
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
        "model": args.model,
        "outputs": {
            "baseline": baseline_csv,
            "method1_single": m1_csv,
            "method2_plan_decode": m2_csv,
        },
        "metrics": {
            "baseline": str(metrics_dir / "baseline.json"),
            "method1_single": str(metrics_dir / "method1_single.json"),
            "method2_plan_decode": str(metrics_dir / "method2_plan_decode.json"),
        },
    }

    with open(base_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStudy completed: {base_dir}")


if __name__ == "__main__":
    main()
