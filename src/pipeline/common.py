import random
import re
from typing import List

from datasets import load_dataset

DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"
VALID_SPLITS = {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST}


def is_cnn_article(article: str) -> bool:
    text = (article or "").lstrip()
    return text.startswith("(CNN)")


def load_cnn_split(split: str):
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got: {split}")

    ds = load_dataset(DATASET_NAME, DATASET_VERSION, split=split)
    ds = ds.filter(lambda x: is_cnn_article(x["article"]))
    return ds


def load_train_split():
    return load_cnn_split(SPLIT_TRAIN)


def load_val_split():
    return load_cnn_split(SPLIT_VAL)


def load_test_split():
    return load_cnn_split(SPLIT_TEST)


def build_summarization_prompt(article: str) -> str:
    return (
        "Summarize the following news article in 3-4 concise factual sentences.\n\n"
        f"Article: {article}\n"
        "Summary:"
    )


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [c.strip() for c in chunks if c.strip()]


def build_plan_from_reference(reference: str, max_points: int = 3) -> str:
    sents = split_sentences(reference)
    if not sents:
        return "- Main event\n- Key detail\n- Outcome"

    picked = sents[:max_points]
    while len(picked) < max_points:
        picked.append(picked[-1])

    return "\n".join([f"- {s}" for s in picked[:max_points]])


def corrupt_summary_text(summary: str, noise_ratio: float = 0.25, seed: int = 42) -> str:
    rng = random.Random(seed)
    words = summary.split()
    if len(words) < 8:
        return summary

    n_drop = max(1, int(len(words) * noise_ratio))
    idxs = list(range(len(words)))
    rng.shuffle(idxs)
    drop_set = set(idxs[:n_drop])
    kept = [w for i, w in enumerate(words) if i not in drop_set]

    # small local shuffle to simulate structural noise
    if len(kept) > 12:
        start = rng.randint(0, len(kept) - 6)
        end = min(len(kept), start + 5)
        seg = kept[start:end]
        rng.shuffle(seg)
        kept[start:end] = seg

    return " ".join(kept)
