import argparse
import inspect
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
    load_cnn_split,
)
from src.dlm.paper_aligned_diffusion import paper_build_prompt, paper_corrupt_text


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train the Method 1 DLM refiner. The --model argument is the "
            "seq2seq denoising backbone; diffusion behavior is implemented by "
            "the timestep corruption/denoising objective."
        )
    )
    p.add_argument(
        "--model",
        default="google/flan-t5-base",
        help="Seq2seq denoising backbone for the DLM refiner, not the LLM draft generator.",
    )
    p.add_argument("--train-split", default=SPLIT_TRAIN)
    p.add_argument("--val-split", default=SPLIT_VAL)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-source-length", type=int, default=1024)
    p.add_argument("--max-target-length", type=int, default=192)
    p.add_argument("--paper-mode", choices=["diffuseq", "seqdiffuseq"], default="diffuseq")
    p.add_argument("--diffusion-steps", type=int, default=8)
    p.add_argument("--output-dir", default="outputs/models/dlm_refiner")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train_ds = load_cnn_split(args.train_split)
    val_ds = load_cnn_split(args.val_split)

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    def preprocess(batch):
        sources = []
        targets = []
        for i, (article, ref) in enumerate(zip(batch["article"], batch["highlights"])):
            t = (i % args.diffusion_steps) + 1
            noisy = paper_corrupt_text(
                text=ref,
                t=t,
                total_steps=args.diffusion_steps,
                mode=args.paper_mode,
                seed=args.seed + i,
            )
            self_cond = ref if args.paper_mode == "seqdiffuseq" and (i % 2 == 0) else ""
            sources.append(
                paper_build_prompt(
                    article=article,
                    noisy=noisy,
                    t=t,
                    mode=args.paper_mode,
                    self_cond=self_cond,
                )
            )
            targets.append(ref)

        model_inputs = tokenizer(sources, truncation=True, max_length=args.max_source_length)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, max_length=args.max_target_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    targs = _build_training_args(Seq2SeqTrainingArguments,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        
        logging_steps=20,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = _build_trainer(
        Seq2SeqTrainer,
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DLM refiner model to {args.output_dir}")


if __name__ == "__main__":
    main()
