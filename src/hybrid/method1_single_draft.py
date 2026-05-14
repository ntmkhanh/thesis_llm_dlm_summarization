from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dlm.paper_aligned_diffusion import DiffusionConfig, PaperAlignedDLMRefiner
from src.pipeline.common import build_summarization_prompt


@dataclass
class Method1SingleDraftOutput:
    article: str
    reference: str
    draft: str
    summary: str
    draft_mode: str = "single"


class Method1SingleDraftHybridModel:
    """Method 1 hybrid model: one LLM draft followed by DLM refinement.

    This class is the explicit hybrid model used by the thesis Single-Draft
    pipeline. It keeps the two roles separate:
    - LLM: generate the initial summary draft.
    - DLM: iteratively denoise/refine that draft into the final summary.
    """

    def __init__(
        self,
        llm_model_dir: str,
        dlm_model_dir: str,
        paper_mode: str = "diffuseq",
        diffusion_steps: int = 8,
        max_new_tokens: int = 180,
    ):
        self.max_new_tokens = max_new_tokens

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_dir)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_dir, device_map="auto")

        self.dlm_refiner = PaperAlignedDLMRefiner(
            model_name_or_path=dlm_model_dir,
            config=DiffusionConfig(
                mode=paper_mode,
                num_steps=diffusion_steps,
                max_new_tokens=max_new_tokens,
            ),
        )

    def generate_draft(self, article: str) -> str:
        prompt = build_summarization_prompt(article)
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.llm.device)

        with torch.no_grad():
            out = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.pad_token_id,
            )

        text = self.llm_tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("Summary:")[-1].strip()

    def refine_draft(self, article: str, draft: str) -> str:
        return self.dlm_refiner.refine(article, draft)

    def summarize(self, article: str, reference: str = "") -> Method1SingleDraftOutput:
        draft = self.generate_draft(article)
        summary = self.refine_draft(article, draft)
        return Method1SingleDraftOutput(
            article=article,
            reference=reference,
            draft=draft,
            summary=summary,
        )
