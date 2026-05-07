import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.genie_core import GenieDiffusionConfig, GenieGaussianDiffusion
from src.dlm.genie_denoiser import GenieDenoiser
from src.pipeline.common import SPLIT_TRAIN, SPLIT_VAL, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Train Method2 pure latent diffusion (article -> latent summary)")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--train-split", default=SPLIT_TRAIN)
    p.add_argument("--val-split", default=SPLIT_VAL)
    p.add_argument("--max-train-samples", type=int, default=3000)
    p.add_argument("--max-val-samples", type=int, default=300)
    p.add_argument("--max-source-len", type=int, default=1024)
    p.add_argument("--max-target-len", type=int, default=192)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--denoiser-layers", type=int, default=4)
    p.add_argument("--denoiser-heads", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", default="outputs/models/method2_latent")
    return p.parse_args()


def batch_iter(ds, batch_size):
    buf = []
    for ex in ds:
        buf.append(ex)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def encode(backbone, tokenizer, texts, max_len, device):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc = backbone.get_encoder()(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], return_dict=True)
    return enc.last_hidden_state


def run_epoch(ds, training, backbone, tokenizer, denoiser, diffusion, opt, args, device):
    denoiser.train(training)
    total = 0.0
    steps = 0
    pbar = tqdm(batch_iter(ds, args.batch_size), total=max(1, len(ds) // args.batch_size), desc=("train" if training else "val"))

    for batch in pbar:
        src = [x["article"] for x in batch]
        tgt = [x["highlights"] for x in batch]

        h_s = encode(backbone, tokenizer, src, args.max_source_len, device)
        x0 = encode(backbone, tokenizer, tgt, args.max_target_len, device)

        b = x0.shape[0]
        t = torch.randint(0, args.timesteps, (b,), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)

        eps_hat = denoiser(x_t, t, h_s)
        loss = F.mse_loss(eps_hat, noise)

        if training:
            opt.zero_grad()
            loss.backward()
            opt.step()

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

    for p in backbone.parameters():
        p.requires_grad = False

    hidden = backbone.config.d_model
    denoiser = GenieDenoiser(hidden, num_layers=args.denoiser_layers, num_heads=args.denoiser_heads).to(device)
    diffusion = GenieGaussianDiffusion(GenieDiffusionConfig(timesteps=args.timesteps), device=device)
    opt = torch.optim.Adam(denoiser.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for ep in range(args.epochs):
        tr = run_epoch(train_ds, True, backbone, tokenizer, denoiser, diffusion, opt, args, device)
        with torch.no_grad():
            va = run_epoch(val_ds, False, backbone, tokenizer, denoiser, diffusion, opt, args, device)
        print(f"epoch={ep+1} train_loss={tr:.4f} val_loss={va:.4f}")

        if va < best_val:
            best_val = va
            best_state = denoiser.state_dict()

    if best_state is None:
        best_state = denoiser.state_dict()

    torch.save(
        {
            "denoiser": best_state,
            "config": {
                "model": args.model,
                "timesteps": args.timesteps,
                "max_source_len": args.max_source_len,
                "max_target_len": args.max_target_len,
                "hidden": hidden,
                "denoiser_layers": args.denoiser_layers,
                "denoiser_heads": args.denoiser_heads,
            },
        },
        os.path.join(args.output_dir, "method2_latent_denoiser.pt"),
    )
    tokenizer.save_pretrained(args.output_dir)
    backbone.config.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"Saved Method2 latent model to {args.output_dir}")


if __name__ == "__main__":
    main()
