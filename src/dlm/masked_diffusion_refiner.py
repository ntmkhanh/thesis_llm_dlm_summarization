import math
import random
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class MaskedDiffusionRefiner:
    """Discrete diffusion-style text refiner via iterative mask-and-denoise.

    This is a practical DLM-style module for Method 1:
    - Add noise by masking a subset of draft tokens
    - Denoise with a seq2seq denoiser
    - Repeat with decreasing mask ratio (coarse -> fine)
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        steps: int = 4,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 192,
        seed: int = 42,
    ):
        self.steps = max(1, steps)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    def _mask_words(self, text: str, mask_ratio: float) -> str:
        words = text.split()
        if len(words) < 6:
            return text

        n_mask = max(1, math.floor(len(words) * mask_ratio))
        idxs = list(range(len(words)))
        self.rng.shuffle(idxs)
        mask_set = set(idxs[:n_mask])

        out = []
        last_mask = False
        for i, w in enumerate(words):
            if i in mask_set:
                if not last_mask:
                    out.append("<extra_id_0>")
                last_mask = True
            else:
                out.append(w)
                last_mask = False
        return " ".join(out)

    def _denoise_once(self, article: str, noisy_summary: str) -> str:
        prompt = (
            "You are a denoising diffusion text refiner for summarization. "
            "Recover a coherent, factual 3-4 sentence summary from the noisy draft.\n\n"
            f"Article: {article}\n\n"
            f"Noisy Draft: {noisy_summary}\n\n"
            "Refined Summary:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=4,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "Refined Summary:" in text:
            return text.split("Refined Summary:", 1)[-1].strip()
        return text.strip()

    def refine(self, article: str, draft: str) -> str:
        current = draft
        for step in range(self.steps):
            # High noise at first, lower later
            ratio = 0.35 - (0.25 * step / max(1, self.steps - 1))
            noisy = self._mask_words(current, max(0.08, ratio))
            current = self._denoise_once(article, noisy)
        return current
