from __future__ import annotations

import json
import sys
from pathlib import Path
import os
import importlib
import argparse

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_config_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()

    module_name = args.config or os.environ.get("TOX_CONFIG") or "config"
    return importlib.import_module(module_name)


config = _load_config_module()
from src.attention import AttentionPooling  # noqa: F401


def load_tokenizer_and_labels():
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
        (config.ARTIFACTS_DIR / "tokenizer.json").read_text(encoding="utf-8")
    )
    labels = json.loads((config.ARTIFACTS_DIR / "labels.json").read_text(encoding="utf-8"))
    return tokenizer, labels


def predict_texts(model, tokenizer, texts: list[str]):
    seq = tokenizer.texts_to_sequences(texts)
    x = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=config.MAX_LEN, padding="post", truncating="post"
    )
    probs = model.predict(x)
    return probs


def main():
    model = tf.keras.models.load_model(
        config.ARTIFACTS_DIR / "model.keras",
        custom_objects={"AttentionPooling": AttentionPooling},
    )
    tokenizer, labels = load_tokenizer_and_labels()

    samples = [
        "I love this!",
        "You are an idiot and I hate you.",
        "I will kill you.",
    ]

    probs = predict_texts(model, tokenizer, samples)
    for text, p in zip(samples, probs):
        top = sorted(zip(labels, p.tolist()), key=lambda t: t[1], reverse=True)
        print("\nTEXT:", text)
        for lbl, score in top:
            print(f"{lbl:14s}: {score:.4f}")


if __name__ == "__main__":
    main()
