import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate metrics from one experiment folder")
    p.add_argument("--exp-dir", required=True, help="Path like outputs/experiments/<exp_id>")
    p.add_argument("--output-csv", default="")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    metrics_dir = exp_dir / "metrics"
    manifest_path = exp_dir / "manifest.json"
    timing = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        timing = manifest.get("timing_seconds", {})

    rows = []
    for name in [
        "baseline_llm",
        "method1_single",
        "method1_multi_refine_each",
        "method1_multi_aggregate_mean",
        "method1_multi_aggregate_learned",
        "method1_llm_dlm",
        "method2_dlm_llm",
    ]:
        path = metrics_dir / f"{name}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        infer_key = f"infer_{name}_sec"
        eval_key = f"eval_{name}_sec"
        rows.append(
            {
                "method": name,
                "infer_seconds": timing.get(infer_key),
                "eval_seconds": timing.get(eval_key),
                **data,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No metrics JSON found in {metrics_dir}")

    df = pd.DataFrame(rows)
    show_cols = [
        "method", "n_samples", "rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1", "infer_seconds"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].sort_values("method").to_string(index=False))

    output_csv = args.output_csv or str(exp_dir / "results_table.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved aggregated table to {output_csv}")


if __name__ == "__main__":
    main()
