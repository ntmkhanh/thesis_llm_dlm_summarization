import argparse
import inspect
import json
import os
import sys
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.common import (
    SPLIT_MODE_NATIVE,
    SPLIT_MODE_CNN_70_20_10,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_summarization_prompt,
    load_cnn_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train LLM SFT for CNN summarization")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--train-split", default=SPLIT_TRAIN)
    p.add_argument("--val-split", default=SPLIT_VAL)
    p.add_argument("--split-mode", choices=[SPLIT_MODE_NATIVE, SPLIT_MODE_CNN_70_20_10], default=SPLIT_MODE_NATIVE)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=3000)
    p.add_argument("--max-val-samples", type=int, default=300)
    p.add_argument("--max-length", type=int, default=1104, help="Legacy total cap; source/target lengths are controlled separately.")
    p.add_argument("--max-source-length", type=int, default=1024)
    p.add_argument("--max-target-length", type=int, default=80)
    p.add_argument("--output-dir", default="outputs/models/llm_sft")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--tuning-mode", choices=["full", "lora", "qlora"], default="full")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return p.parse_args()


def format_example(article: str, summary: str) -> str:
    prompt = build_summarization_prompt(article)
    return f"{prompt} {summary}"


def build_sft_example(tokenizer, article: str, summary: str, max_source_length: int, max_target_length: int):
    prompt = build_summarization_prompt(article)
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_source_length,
        add_special_tokens=True,
    )["input_ids"]

    target_max = max(1, max_target_length - 1)
    target_ids = tokenizer(
        " " + str(summary),
        truncation=True,
        max_length=target_max,
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def build_causal_sft_collator(tokenizer):
    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate



def _build_training_args(args_class, **kwargs):
    sig = inspect.signature(args_class.__init__)
    supported = set(sig.parameters.keys())
    out = {k: v for k, v in kwargs.items() if k in supported}
    if "evaluation_strategy" in supported:
        out["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        out["eval_strategy"] = "epoch"
    if "save_strategy" in supported:
        out["save_strategy"] = "epoch"
    return args_class(**out)


def _build_trainer(trainer_class, **kwargs):
    sig = inspect.signature(trainer_class.__init__)
    supported = set(sig.parameters.keys())
    out = {k: v for k, v in kwargs.items() if k in supported}
    return trainer_class(**out)


def _maybe_build_quant_config(mode: str) -> Optional[BitsAndBytesConfig]:
    if mode != "qlora":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _apply_lora(model, args):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise RuntimeError(
            "LoRA/QLoRA requested but PEFT is missing. Install with: pip install peft"
        ) from e

    if args.tuning_mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_cnn_split(args.train_split, split_mode=args.split_mode, seed=args.split_seed)
    val_ds = load_cnn_split(args.val_split, split_mode=args.split_mode, seed=args.split_seed)

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    def preprocess(batch):
        rows = [
            build_sft_example(
                tokenizer,
                article,
                summary,
                args.max_source_length,
                args.max_target_length,
            )
            for article, summary in zip(batch["article"], batch["highlights"])
        ]
        return {
            "input_ids": [row["input_ids"] for row in rows],
            "attention_mask": [row["attention_mask"] for row in rows],
            "labels": [row["labels"] for row in rows],
        }

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    quant_config = _maybe_build_quant_config(args.tuning_mode)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto" if args.tuning_mode == "qlora" else None,
    )
    if args.tuning_mode in ("lora", "qlora"):
        model = _apply_lora(model, args)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    targs = _build_training_args(TrainingArguments,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        
        logging_steps=20,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    collator = build_causal_sft_collator(tokenizer)

    trainer = _build_trainer(
        Trainer,
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    log_history = trainer.state.log_history
    train_loss_history = [
        item for item in log_history
        if "loss" in item and "eval_loss" not in item
    ]
    eval_loss_history = [
        item for item in log_history
        if "eval_loss" in item
    ]
    with open(os.path.join(args.output_dir, "trainer_log_history.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "train_loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_loss_history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "eval_loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(eval_loss_history, f, indent=2, ensure_ascii=False)

    best_info = {
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    with open(os.path.join(args.output_dir, "best_checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved best LLM SFT model to {args.output_dir}")
    print(f"Best checkpoint metadata: {os.path.join(args.output_dir, 'best_checkpoint.json')}")
    print(f"Trainer log history: {os.path.join(args.output_dir, 'trainer_log_history.json')}")
    print(f"Train loss history: {os.path.join(args.output_dir, 'train_loss_history.json')}")
    print(f"Eval loss history: {os.path.join(args.output_dir, 'eval_loss_history.json')}")


if __name__ == "__main__":
    main()
