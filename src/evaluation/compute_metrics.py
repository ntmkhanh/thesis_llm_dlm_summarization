import argparse
import json

import evaluate
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ROUGE and BERTScore from summary CSV")
    parser.add_argument("--input", default="outputs/drafts/baseline_qwen.csv")
    parser.add_argument("--pred-col", default="summary")
    parser.add_argument("--ref-col", default="reference")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return value


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    predictions = df[args.pred_col].fillna("").tolist()
    references = df[args.ref_col].fillna("").tolist()

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bert = bertscore.compute(predictions=predictions, references=references, lang=args.lang)

    results = {
        "n_samples": len(df),
        "rouge1": _to_float(rouge_scores.get("rouge1", 0.0)),
        "rouge2": _to_float(rouge_scores.get("rouge2", 0.0)),
        "rougeL": _to_float(rouge_scores.get("rougeL", 0.0)),
        "rougeLsum": _to_float(rouge_scores.get("rougeLsum", 0.0)),
        "bertscore_precision": sum(bert["precision"]) / len(bert["precision"]),
        "bertscore_recall": sum(bert["recall"]) / len(bert["recall"]),
        "bertscore_f1": sum(bert["f1"]) / len(bert["f1"]),
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
