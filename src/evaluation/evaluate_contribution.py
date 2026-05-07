import argparse
import json
from pathlib import Path

import evaluate
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compare methods and report contribution deltas vs baseline")
    p.add_argument("--baseline", required=True, help="CSV path baseline")
    p.add_argument("--candidates", nargs="+", required=True, help="CSV paths to compare")
    p.add_argument("--pred-col", default="summary")
    p.add_argument("--ref-col", default="reference")
    p.add_argument("--article-col", default="article")
    p.add_argument("--lang", default="en")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-csv", default="")
    return p.parse_args()


def token_f1(pred: str, ref: str) -> float:
    p = (pred or "").lower().split()
    r = (ref or "").lower().split()
    if not p or not r:
        return 0.0
    rc = {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    m = 0
    for t in p:
        if rc.get(t, 0) > 0:
            m += 1
            rc[t] -= 1
    prec = m / len(p)
    rec = m / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def factual_proxy(df, pred_col: str, article_col: str) -> float:
    vals = [token_f1(p, a) for p, a in zip(df[pred_col].fillna(""), df[article_col].fillna(""))]
    return sum(vals) / max(1, len(vals))


def length_stats(df, pred_col: str):
    lengths = [len(str(x).split()) for x in df[pred_col].fillna("")]
    return sum(lengths) / max(1, len(lengths))


def latency_stats(df):
    if "latency_sec" not in df.columns:
        return None
    vals = [float(x) for x in df["latency_sec"].fillna(0.0)]
    return sum(vals) / max(1, len(vals))


def compute_metrics(df, pred_col, ref_col, article_col, rouge_metric, bert_metric, lang):
    preds = df[pred_col].fillna("").tolist()
    refs = df[ref_col].fillna("").tolist()
    r = rouge_metric.compute(predictions=preds, references=refs)
    b = bert_metric.compute(predictions=preds, references=refs, lang=lang)
    return {
        "n_samples": len(df),
        "rouge1": float(r.get("rouge1", 0.0)),
        "rouge2": float(r.get("rouge2", 0.0)),
        "rougeL": float(r.get("rougeL", 0.0)),
        "rougeLsum": float(r.get("rougeLsum", 0.0)),
        "bertscore_f1": float(sum(b["f1"]) / max(1, len(b["f1"]))),
        "factual_proxy_f1": float(factual_proxy(df, pred_col, article_col)),
        "avg_summary_len": float(length_stats(df, pred_col)),
        "avg_latency_sec": latency_stats(df),
    }


def add_delta(row, base):
    out = dict(row)
    for k in ["rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1", "factual_proxy_f1"]:
        out[f"delta_{k}"] = row[k] - base[k]
    if row.get("avg_latency_sec") is not None and base.get("avg_latency_sec") is not None:
        out["delta_avg_latency_sec"] = row["avg_latency_sec"] - base["avg_latency_sec"]
    else:
        out["delta_avg_latency_sec"] = None
    return out


def main():
    args = parse_args()
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    base_df = pd.read_csv(args.baseline)
    base_metrics = compute_metrics(base_df, args.pred_col, args.ref_col, args.article_col, rouge, bert, args.lang)

    rows = []
    base_name = Path(args.baseline).stem
    rows.append({"method": base_name, **base_metrics})

    for c in args.candidates:
        df = pd.read_csv(c)
        m = compute_metrics(df, args.pred_col, args.ref_col, args.article_col, rouge, bert, args.lang)
        row = {"method": Path(c).stem, **m}
        row = add_delta(row, base_metrics)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))

    if args.output_csv:
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved table to {args.output_csv}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Saved json to {args.output_json}")


if __name__ == "__main__":
    main()
