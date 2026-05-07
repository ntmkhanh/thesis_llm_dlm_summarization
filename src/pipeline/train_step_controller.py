import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dlm.step_controller import StepController, StepControllerConfig, save_step_controller
from src.pipeline.common import SPLIT_TRAIN, build_summarization_prompt, load_cnn_split


def parse_args():
    p = argparse.ArgumentParser(description="Train adaptive step controller for GENIE")
    p.add_argument("--llm-model-dir", required=True)
    p.add_argument("--genie-model-dir", required=True)
    p.add_argument("--split", default=SPLIT_TRAIN)
    p.add_argument("--max-samples", type=int, default=3000)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--step-bins", default="20,40,60,80,100")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="outputs/models/genie/step_controller.pt")
    return p.parse_args()


def overlap_f1(pred: str, ref: str) -> float:
    p = pred.lower().split()
    r = ref.lower().split()
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


def draft_quality_to_bin(score: float, bins):
    # lower quality -> more steps
    # simple curriculum target
    if score < 0.20:
        return len(bins) - 1
    if score < 0.28:
        return max(0, len(bins) - 2)
    if score < 0.36:
        return max(0, len(bins) - 3)
    if score < 0.44:
        return max(0, len(bins) - 4)
    return 0


def encode(backbone, tokenizer, texts, max_len, device):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc = backbone.get_encoder()(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], return_dict=True)
    return enc.last_hidden_state.mean(dim=1)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    bins = [int(x.strip()) for x in args.step_bins.split(",") if x.strip()]
    ds = load_cnn_split(args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    llm_tok = AutoTokenizer.from_pretrained(args.llm_model_dir)
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model_dir, device_map="auto")

    tok = AutoTokenizer.from_pretrained(args.genie_model_dir)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(args.genie_model_dir, device_map="auto")
    device = backbone.device

    hidden = backbone.config.d_model
    ctrl = StepController(StepControllerConfig(hidden_size=hidden, step_bins=bins)).to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        ctrl.train()
        total = 0.0
        steps = 0
        for ex in tqdm(ds, total=len(ds), desc=f"train step-controller ep{ep+1}"):
            article = ex["article"]
            ref = ex["highlights"]

            # 1) one-draft generation
            prompt = build_summarization_prompt(article)
            inputs = llm_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llm.device)
            with torch.no_grad():
                out = llm.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=llm_tok.pad_token_id)
            draft = llm_tok.decode(out[0], skip_special_tokens=True).split("Summary:")[-1].strip()

            # 2) pseudo target step bin from draft quality
            q = overlap_f1(draft, ref)
            y = torch.tensor([draft_quality_to_bin(q, bins)], device=device)

            # 3) predict
            za = encode(backbone, tok, [article], 1024, device)
            zd = encode(backbone, tok, [draft], 192, device)
            logits = ctrl(za, zd)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1
            if steps % 100 == 0:
                print({"epoch": ep + 1, "step": steps, "loss": total / steps})

    save_step_controller(ctrl.eval(), args.output)
    print(f"Saved step controller to {args.output}")


if __name__ == "__main__":
    main()
