from datasets import load_dataset


def _is_cnn_article(article: str) -> bool:
    text = (article or "").lstrip()
    return text.startswith("(CNN)")


def load_cnn(splits="test[:100]", cnn_only=True):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=splits)
    if cnn_only:
        dataset = dataset.filter(lambda x: _is_cnn_article(x["article"]))
    return dataset

if __name__ == "__main__":  
    ds = load_cnn()
    print(ds[0].keys())
    print(ds[0]["article"][:500])
    print(ds[0]["highlights"])
    print(len(ds))
