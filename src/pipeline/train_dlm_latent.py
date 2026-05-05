import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import csv
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.core_diffusion import DiffusionSchedule
from src.dlm.latent_denoiser import LatentDenoiser
from src.pipeline.common import SPLIT_TRAIN, SPLIT_VAL, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Train latent diffusion (core_diffusion + latent_denoiser)")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--train-split", default=SPLIT_TRAIN)
    p.add_argument("--val-split", default=SPLIT_VAL)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)
    p.add_argument("--max-source-len", type=int, default=1024)
    p.add_argument("--max-target-len", type=int, default=192)
    p.add_argument("--self-condition", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--top-k-checkpoints", type=int, default=3)
    p.add_argument("--output-dir", default="outputs/models/dlm_latent")
    return p.parse_args()


def batch_iter(dataset, batch_size):
    buf = []
    for ex in dataset:
        buf.append(ex)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def encode(backbone, tokenizer, texts, max_len, device):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc = backbone.get_encoder()(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            return_dict=True,
        )
    return enc.last_hidden_state


def train_or_eval_epoch(ds, training, backbone, tokenizer, denoiser, schedule, optim, args, device):
    denoiser.train(training)
    total = 0.0
    steps = 0

    iterator = batch_iter(ds, args.batch_size)
    pbar = tqdm(iterator, total=max(1, len(ds) // args.batch_size), desc=("train" if training else "val"))
    for batch in pbar:
        articles = [x["article"] for x in batch]
        refs = [x["highlights"] for x in batch]

        z_article = encode(backbone, tokenizer, articles, args.max_source_len, device)
        z0 = encode(backbone, tokenizer, refs, args.max_target_len, device)

        b = z0.shape[0]
        t = torch.randint(0, args.timesteps, (b,), device=device)
        noise = torch.randn_like(z0)
        zt = schedule.q_sample(z0, t, noise)

        article_ctx = z_article.mean(dim=1)
        self_cond = None

        if args.self_condition and torch.rand(()) < 0.5:
            with torch.no_grad():
                eps_prev = denoiser(zt, article_ctx, t, self_cond=None)
                self_cond = schedule.predict_z0_from_eps(zt, t, eps_prev)

        eps_hat = denoiser(zt, article_ctx, t, self_cond=self_cond)
        loss = F.mse_loss(eps_hat, noise)

        if training:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total += float(loss.item())
        steps += 1
        if steps % 10 == 0:
            pbar.set_postfix(loss=total / steps)

    return total / max(1, steps)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = load_cnn_split(args.train_split)
    val_ds = load_cnn_split(args.val_split)
    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto")
    device = backbone.device

    # freeze backbone for stable latent diffusion training
    for p in backbone.parameters():
        p.requires_grad = False

    hidden = backbone.config.d_model
    denoiser = LatentDenoiser(hidden, self_condition=args.self_condition).to(device)
    schedule = DiffusionSchedule(args.timesteps, args.beta_start, args.beta_end, device=str(device))
    optim = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    history = []
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    best_ckpts = []  # list of (val_loss, epoch, path)

    for ep in range(args.epochs):
        train_loss = train_or_eval_epoch(train_ds, True, backbone, tokenizer, denoiser, schedule, optim, args, device)
        with torch.no_grad():
            val_loss = train_or_eval_epoch(val_ds, False, backbone, tokenizer, denoiser, schedule, optim, args, device)
        print(f"epoch={ep+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        history.append({"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss})

        ckpt = {
            "denoiser": denoiser.state_dict(),
            "config": {
                "model": args.model,
                "timesteps": args.timesteps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "max_source_len": args.max_source_len,
                "max_target_len": args.max_target_len,
                "self_condition": args.self_condition,
                "hidden": hidden,
            },
            "epoch": ep + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        if args.save_every_epoch:
            ep_path = os.path.join(
                args.output_dir,
                f"latent_denoiser_epoch{ep+1}_val{val_loss:.6f}.pt",
            )
            torch.save(ckpt, ep_path)

        if val_loss < best_val - args.early_stop_min_delta:
            best_val = val_loss
            best_epoch = ep + 1
            no_improve = 0
            # Canonical best checkpoint path (always points to current best)
            torch.save(ckpt, os.path.join(args.output_dir, "latent_denoiser.pt"))
        else:
            no_improve += 1

        # Save epoch checkpoint with val_loss in filename, then keep only top-k by val loss.
        scored_path = os.path.join(
            args.output_dir,
            f"best_epoch{ep+1}_val{val_loss:.6f}.pt",
        )
        torch.save(ckpt, scored_path)
        best_ckpts.append((val_loss, ep + 1, scored_path))
        best_ckpts.sort(key=lambda x: x[0])  # lower val_loss is better
        if len(best_ckpts) > args.top_k_checkpoints:
            to_remove = best_ckpts[args.top_k_checkpoints:]
            best_ckpts = best_ckpts[: args.top_k_checkpoints]
            for _, _, rm_path in to_remove:
                if os.path.exists(rm_path):
                    os.remove(rm_path)

        if no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {ep+1} (best epoch: {best_epoch}, best val: {best_val:.4f})")
            break

    # If early-stop never improved due to edge case, fallback save final.
    if not os.path.exists(os.path.join(args.output_dir, "latent_denoiser.pt")):
        torch.save(
            {
                "denoiser": denoiser.state_dict(),
                "config": {
                    "model": args.model,
                    "timesteps": args.timesteps,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "max_source_len": args.max_source_len,
                    "max_target_len": args.max_target_len,
                    "self_condition": args.self_condition,
                    "hidden": hidden,
                },
                "epoch": len(history),
                "train_loss": history[-1]["train_loss"] if history else None,
                "val_loss": history[-1]["val_loss"] if history else None,
            },
            os.path.join(args.output_dir, "latent_denoiser.pt"),
        )

    # Save loss history as JSON and CSV.
    with open(os.path.join(args.output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "loss_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    # Plot loss curves.
    if history:
        xs = [h["epoch"] for h in history]
        train_ys = [h["train_loss"] for h in history]
        val_ys = [h["val_loss"] for h in history]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, train_ys, marker="o", label="Train Loss")
        plt.plot(xs, val_ys, marker="s", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Latent Denoiser Training Curve")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=160)
        plt.close()
    tokenizer.save_pretrained(args.output_dir)
    backbone.config.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        meta = vars(args).copy()
        meta["best_epoch"] = best_epoch
        meta["best_val_loss"] = best_val
        meta["top_k_kept"] = [
            {"epoch": ep, "val_loss": vl, "path": path}
            for (vl, ep, path) in best_ckpts
        ]
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved latent model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
