import math
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.dlm.latent_fusion_gating import LatentFusionGating


@dataclass
class DiffusionConfig:
    mode: str = "seqdiffuseq"  # diffuseq | seqdiffuseq
    num_steps: int = 8
    max_source_len: int = 1024
    max_new_tokens: int = 192
    seed: int = 42


class PaperAlignedDLMRefiner:
    """Paper-aligned lightweight diffusion refiner.

    - DiffuSeq-style: timestep-conditioned corruption + iterative denoise.
    - SeqDiffuSeq-style: adds self-conditioning text and position-adaptive noise.

    This is a practical thesis implementation inspired by paper mechanisms,
    while still using HF seq2seq backbones for trainability.
    """

    def __init__(self, model_name_or_path: str, config: DiffusionConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")

    def _t_to_ratio(self, t: int) -> float:
        # high noise at early reverse steps, lower noise later
        # t in [1..T]
        T = max(1, self.config.num_steps)
        x = t / T
        return 0.08 + 0.35 * x

    def _adaptive_position_mask(self, words, ratio: float):
        n = len(words)
        if n < 6:
            return words

        keep = []
        for i, w in enumerate(words):
            pos = i / max(1, n - 1)
            # SeqDiffuSeq-inspired adaptive schedule: later positions slightly noisier
            p = ratio * (0.8 + 0.4 * pos)
            if self.rng.random() < p:
                keep.append("<extra_id_0>")
            else:
                keep.append(w)

        # collapse consecutive masks
        collapsed = []
        for tok in keep:
            if tok == "<extra_id_0>" and collapsed and collapsed[-1] == "<extra_id_0>":
                continue
            collapsed.append(tok)
        return collapsed

    def corrupt(self, text: str, t: int) -> str:
        ratio = self._t_to_ratio(t)
        words = text.split()
        if self.config.mode == "seqdiffuseq":
            noisy = self._adaptive_position_mask(words, ratio)
            return " ".join(noisy)

        # DiffuSeq-like uniform corruption
        if len(words) < 6:
            return text
        n_mask = max(1, math.floor(len(words) * ratio))
        ids = list(range(len(words)))
        self.rng.shuffle(ids)
        mset = set(ids[:n_mask])
        out = []
        last = False
        for i, w in enumerate(words):
            if i in mset:
                if not last:
                    out.append("<extra_id_0>")
                last = True
            else:
                out.append(w)
                last = False
        return " ".join(out)

    def _build_prompt(self, article: str, noisy: str, t: int, self_cond: str = "") -> str:
        if self.config.mode == "seqdiffuseq":
            return (
                f"Denoise summary at timestep t={t} with self-conditioning.\n\n"
                f"Article: {article}\n\n"
                f"Noisy Summary: {noisy}\n\n"
                f"Self-Conditioned Previous Prediction: {self_cond}\n\n"
                "Refined Summary:"
            )

        return (
            f"Denoise summary at timestep t={t}.\n\n"
            f"Article: {article}\n\n"
            f"Noisy Summary: {noisy}\n\n"
            "Refined Summary:"
        )

    def denoise_once(self, article: str, noisy: str, t: int, self_cond: str = "") -> str:
        prompt = self._build_prompt(article, noisy, t, self_cond=self_cond)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_len,
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=4,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "Refined Summary:" in txt:
            txt = txt.split("Refined Summary:", 1)[-1]
        return txt.strip()

    def refine(self, article: str, draft: str) -> str:
        x = draft
        prev = ""
        for t in range(self.config.num_steps, 0, -1):
            noisy = self.corrupt(x, t)
            x = self.denoise_once(article, noisy, t, self_cond=prev)
            prev = x
        return x

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into a continuous vector (latent proxy).

        NOTE:
        - This is a practical latent-space proxy for thesis experiments.
        - We use encoder hidden-state mean pooling as z representation.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_len,
        ).to(self.model.device)
        with torch.no_grad():
            enc = self.model.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
        h = enc.last_hidden_state  # [1, L, H]
        return h.mean(dim=1).squeeze(0)  # [H]

    def aggregate_drafts_in_latent_mean(self, article: str, drafts: List[str]) -> str:
        """Multi-Draft (ii): latent aggregation before reverse denoising.

        Flow mapping to proposal:
        1) Encode each draft to latent z_i
        2) Aggregate z* = mean(z_i)
        3) Choose nearest draft to z* as initialization
        4) Run diffusion reverse denoising on that initialization
        """
        if not drafts:
            return ""
        if len(drafts) == 1:
            return self.refine(article, drafts[0])

        z_list = [self.encode_text(d) for d in drafts]
        z_star = torch.stack(z_list, dim=0).mean(dim=0)
        sims = [F.cosine_similarity(z.unsqueeze(0), z_star.unsqueeze(0)).item() for z in z_list]
        best_idx = max(range(len(drafts)), key=lambda i: sims[i])
        init_draft = drafts[best_idx]
        return self.refine(article, init_draft)

    def aggregate_drafts_in_latent_learned(
        self,
        article: str,
        drafts: List[str],
        gating: LatentFusionGating,
    ) -> str:
        """Learned latent fusion with trainable gating weights."""
        if not drafts:
            return ""
        if len(drafts) == 1:
            return self.refine(article, drafts[0])

        z_article = self.encode_text(article)
        z_drafts = torch.stack([self.encode_text(d) for d in drafts], dim=0)
        with torch.no_grad():
            z_star, _ = gating.fuse(z_article, z_drafts)
        sims = [F.cosine_similarity(z.unsqueeze(0), z_star.unsqueeze(0)).item() for z in z_drafts]
        best_idx = max(range(len(drafts)), key=lambda i: sims[i])
        init_draft = drafts[best_idx]
        return self.refine(article, init_draft)

    def aggregate_drafts_in_latent(
        self,
        article: str,
        drafts: List[str],
        fusion_mode: str = "mean",
        gating: Optional[LatentFusionGating] = None,
    ) -> str:
        if fusion_mode == "learned":
            if gating is None:
                raise ValueError("fusion_mode='learned' requires a loaded gating model")
            return self.aggregate_drafts_in_latent_learned(article, drafts, gating)
        return self.aggregate_drafts_in_latent_mean(article, drafts)


def paper_noise_ratio(t: int, total_steps: int) -> float:
    total_steps = max(1, total_steps)
    x = t / total_steps
    return 0.08 + 0.35 * x


def paper_corrupt_text(text: str, t: int, total_steps: int, mode: str, seed: int) -> str:
    rng = random.Random(seed + t)
    words = text.split()
    if len(words) < 6:
        return text

    ratio = paper_noise_ratio(t, total_steps)

    if mode == "seqdiffuseq":
        out = []
        n = len(words)
        for i, w in enumerate(words):
            pos = i / max(1, n - 1)
            p = ratio * (0.8 + 0.4 * pos)
            out.append("<extra_id_0>" if rng.random() < p else w)
        merged = []
        for tok in out:
            if tok == "<extra_id_0>" and merged and merged[-1] == "<extra_id_0>":
                continue
            merged.append(tok)
        return " ".join(merged)

    n_mask = max(1, math.floor(len(words) * ratio))
    ids = list(range(len(words)))
    rng.shuffle(ids)
    mset = set(ids[:n_mask])
    out = []
    last = False
    for i, w in enumerate(words):
        if i in mset:
            if not last:
                out.append("<extra_id_0>")
            last = True
        else:
            out.append(w)
            last = False
    return " ".join(out)


def paper_build_prompt(article: str, noisy: str, t: int, mode: str, self_cond: str = "") -> str:
    if mode == "seqdiffuseq":
        return (
            f"Denoise summary at timestep t={t} with self-conditioning.\n\n"
            f"Article: {article}\n\n"
            f"Noisy Summary: {noisy}\n\n"
            f"Self-Conditioned Previous Prediction: {self_cond}\n\n"
            "Refined Summary:"
        )
    return (
        f"Denoise summary at timestep t={t}.\n\n"
        f"Article: {article}\n\n"
        f"Noisy Summary: {noisy}\n\n"
        "Refined Summary:"
    )
