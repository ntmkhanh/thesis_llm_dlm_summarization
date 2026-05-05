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

    rows = []
    for name in ["method1_llm_dlm", "method2_dlm_llm"]:
        path = metrics_dir / f"{name}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({"method": name, **data})

    if not rows:
        raise FileNotFoundError(f"No metrics JSON found in {metrics_dir}")

    df = pd.DataFrame(rows)
    show_cols = [
        "method", "n_samples", "rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].sort_values("method").to_string(index=False))

    output_csv = args.output_csv or str(exp_dir / "results_table.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved aggregated table to {output_csv}")


if __name__ == "__main__":
    main()
