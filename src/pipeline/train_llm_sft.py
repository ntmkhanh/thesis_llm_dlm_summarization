import argparse
import inspect
import os
import sys
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LLM SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
