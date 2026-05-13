import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_summary_prompt(article: str, style: str = "3-4 concise sentences") -> str:
    return (
        f"Summarize the following news article in {style}.\\n\\n"
        f"Article: {article}\\n"
        "Summary:"
    )


def _extract_after_tag(text: str, tag: str) -> str:
    if tag in text:
        return text.split(tag, 1)[-1].strip()
    return text.strip()


def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    adapter_cfg = os.path.join(model_name, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_input_tokens: int = 2048,
    max_new_tokens: int = 160,
    num_beams: int = 1,
    do_sample: bool = False,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def summarize_article(
    tokenizer,
    model,
    article: str,
    style: str = "3-4 concise sentences",
    max_new_tokens: int = 160,
    num_beams: int = 1,
    do_sample: bool = False,
) -> str:
    prompt = build_summary_prompt(article, style=style)
    generated = generate_text(
        tokenizer,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    return _extract_after_tag(generated, "Summary:")


def word_overlap_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    match = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            match += 1
            ref_counts[token] -= 1

    precision = match / len(pred_tokens)
    recall = match / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def select_best_candidate(candidates: List[str], article: str) -> str:
    if not candidates:
        return ""

    best_idx = 0
    best_score = -1.0

    for i, candidate in enumerate(candidates):
        score = word_overlap_f1(candidate, article)
        if score > best_score:
            best_score = score
            best_idx = i

    return candidates[best_idx]


def refine_summary_with_editor(tokenizer, model, article: str, draft: str, max_new_tokens: int = 160) -> str:
    prompt = (
        "You are an expert summary editor. Improve the draft summary by fixing logical errors, "
        "removing repetition, and improving global coherence while preserving key facts.\\n\\n"
        f"Article: {article}\\n\\n"
        f"Draft Summary: {draft}\\n\\n"
        "Refined Summary:"
    )
    generated = generate_text(
        tokenizer,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
    )
    return _extract_after_tag(generated, "Refined Summary:")
