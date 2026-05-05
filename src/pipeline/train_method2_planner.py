import argparse
import os
import sys

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.common import (
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_plan_from_reference,
    load_cnn_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Method2 planner: article -> plan")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--train-split", default=SPLIT_TRAIN)
    p.add_argument("--val-split", default=SPLIT_VAL)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-source-length", type=int, default=1024)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--output-dir", default="outputs/models/method2_planner")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train_ds = load_cnn_split(args.train_split)
    val_ds = load_cnn_split(args.val_split)

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    def preprocess(batch):
        sources = [
            "Create a 3-bullet global summary plan for this article.\n\n"
            f"Article: {a}\n\nPlan:"
            for a in batch["article"]
        ]
        targets = [build_plan_from_reference(h, max_points=3) for h in batch["highlights"]]

        x = tok(sources, truncation=True, max_length=args.max_source_length)
        with tok.as_target_tokenizer():
            y = tok(targets, truncation=True, max_length=args.max_target_length)
        x["labels"] = y["input_ids"]
        return x

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        save_total_limit=2,
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tok, model=model),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved planner model to {args.output_dir}")


if __name__ == "__main__":
    main()
