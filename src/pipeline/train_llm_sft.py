import argparse
import os
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.common import (
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_summarization_prompt,
    load_cnn_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train LLM SFT for CNN summarization")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-split", default=SPLIT_TRAIN)
    p.add_argument("--val-split", default=SPLIT_VAL)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--output-dir", default="outputs/models/llm_sft")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    return p.parse_args()


def format_example(article: str, summary: str) -> str:
    prompt = build_summarization_prompt(article)
    return f"{prompt} {summary}"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_cnn_split(args.train_split)
    val_ds = load_cnn_split(args.val_split)

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    def preprocess(batch):
        texts = [format_example(a, h) for a, h in zip(batch["article"], batch["highlights"])]
        tok = tokenizer(texts, truncation=True, max_length=args.max_length)
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(args.model)

    targs = TrainingArguments(
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
        fp16=False,
        bf16=False,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LLM SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
