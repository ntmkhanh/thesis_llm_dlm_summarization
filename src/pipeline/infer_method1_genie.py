import argparse
import os
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.genie_core import GenieDiffusionConfig, GenieGaussianDiffusion
from src.dlm.genie_denoiser import GenieDenoiser
from src.dlm.genie_grounding import decode_with_backbone
from src.dlm.step_controller import load_step_controller
from src.pipeline.common import SPLIT_TEST, build_summarization_prompt, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Method1 GENIE: LLM draft -> GENIE latent denoise -> grounding decode")
    p.add_argument("--llm-model-dir", required=True)
    p.add_argument("--genie-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TEST)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--adaptive-steps", action="store_true")
    p.add_argument("--step-controller-model", default="")
    p.add_argument("--output", default="outputs/drafts/method1_genie.csv")
    return p.parse_args()


def llm_draft(tokenizer, model, article: str, max_new_tokens: int) -> str:
    prompt = build_summarization_prompt(article)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt.split("Summary:")[-1].strip()


def encode(backbone, tokenizer, texts, max_len, device):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc = backbone.get_encoder()(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], return_dict=True)
    return enc.last_hidden_state


def reverse_genie(article, x_t, backbone, tokenizer, denoiser, diffusion, cfg, device, steps_override=None):
    h_s = encode(backbone, tokenizer, [article], cfg["max_source_len"], device)
    xt = x_t
    n_steps = int(steps_override) if steps_override is not None else int(cfg["timesteps"])
    n_steps = max(1, min(n_steps, int(cfg["timesteps"])))
    for step in reversed(range(n_steps)):
        t = torch.tensor([step], device=device)
        eps = denoiser(xt, t, h_s)
        xt = diffusion.p_step(xt, step, eps)
    return xt


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = load_cnn_split(args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    llm_tok = AutoTokenizer.from_pretrained(args.llm_model_dir)
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model_dir, device_map="auto")

    ckpt = torch.load(os.path.join(args.genie_model_dir, "genie_denoiser.pt"), map_location="cpu")
    cfg = ckpt["config"]

    tok = AutoTokenizer.from_pretrained(args.genie_model_dir)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"], device_map="auto")
    device = backbone.device

    denoiser = GenieDenoiser(cfg["hidden"], num_layers=cfg["denoiser_layers"], num_heads=cfg["denoiser_heads"]).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()

    diffusion = GenieGaussianDiffusion(GenieDiffusionConfig(timesteps=cfg["timesteps"]), device=device)
    step_controller = None
    if args.adaptive_steps:
        if not args.step_controller_model:
            raise ValueError("--adaptive-steps requires --step-controller-model")
        step_controller = load_step_controller(args.step_controller_model, device=str(device)).to(device)

    rows = []
    for ex in tqdm(ds, total=len(ds), desc="Method1 GENIE"):
        article = ex["article"]
        reference = ex["highlights"]

        draft = llm_draft(llm_tok, llm, article, args.max_new_tokens)
        x0 = encode(backbone, tok, [draft], cfg["max_target_len"], device)

        noise = torch.randn_like(x0)
        tmax = torch.tensor([cfg["timesteps"] - 1], device=device)
        x_t = diffusion.q_sample(x0, tmax, noise)

        used_steps = cfg["timesteps"]
        if step_controller is not None:
            z_article = encode(backbone, tok, [article], cfg["max_source_len"], device)
            za = z_article.mean(dim=1)
            zd = x0.mean(dim=1)
            with torch.no_grad():
                used_steps = int(step_controller.predict_steps(za, zd).item())

        x_hat = reverse_genie(
            article,
            x_t,
            backbone,
            tok,
            denoiser,
            diffusion,
            cfg,
            device,
            steps_override=used_steps,
        )
        summary = decode_with_backbone(backbone, tok, x_hat, max_new_tokens=args.max_new_tokens)

        rows.append({"article": article, "reference": reference, "draft": draft, "summary": summary, "used_steps": used_steps})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
