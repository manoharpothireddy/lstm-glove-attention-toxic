from __future__ import annotations

import json
import sys
from pathlib import Path
import os
import importlib
import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

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
from src.data_preprocessing import load_text_and_labels, prepare_tokenizer_and_sequences


def main():
    model = tf.keras.models.load_model(
        config.ARTIFACTS_DIR / "model.keras",
        custom_objects={"AttentionPooling": AttentionPooling},
    )

    labels = json.loads((config.ARTIFACTS_DIR / "labels.json").read_text(encoding="utf-8"))

    texts, y = load_text_and_labels(
        tfds_dataset=config.TFDS_DATASET,
        hf_dataset=config.HF_DATASET,
        hf_split=config.HF_SPLIT,
        labels=labels,
    )

    max_examples = getattr(config, "MAX_TRAIN_EXAMPLES", None)
    if isinstance(max_examples, int) and max_examples > 0:
        texts = texts[:max_examples]
        y = y[:max_examples]

    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
        (config.ARTIFACTS_DIR / "tokenizer.json").read_text(encoding="utf-8")
    )

    prepared = prepare_tokenizer_and_sequences(
        texts=texts,
        y=y,
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_LEN,
        val_fraction=0.1,
        seed=config.RANDOM_SEED,
    )
    prepared = prepared.__class__(
        x_train=prepared.x_train,
        y_train=prepared.y_train,
        x_val=prepared.x_val,
        y_val=prepared.y_val,
        tokenizer=tokenizer,
    )

    probs = model.predict(prepared.x_val, batch_size=config.BATCH_SIZE)
    preds = (probs >= 0.5).astype(int)

    print("AUC macro:", roc_auc_score(prepared.y_val, probs, average="macro"))
    print(classification_report(prepared.y_val, preds, target_names=labels, zero_division=0))


if __name__ == "__main__":
    main()
