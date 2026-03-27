from __future__ import annotations

import sys
import json
from pathlib import Path
import urllib.request
import zipfile
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
from src.data_preprocessing import (
    build_embedding_matrix,
    load_text_and_labels,
    prepare_tokenizer_and_sequences,
)
from src.model import build_model


def ensure_glove_file():
    if config.GLOVE_FILE.exists():
        return

    if not getattr(config, "AUTO_DOWNLOAD_GLOVE", False):
        raise FileNotFoundError(
            f"Missing GloVe file: {config.GLOVE_FILE}. "
            "Set AUTO_DOWNLOAD_GLOVE=True in config.py or place the file manually."
        )

    url = getattr(config, "GLOVE_ZIP_URL", None)
    if not url:
        raise ValueError("AUTO_DOWNLOAD_GLOVE is enabled but GLOVE_ZIP_URL is not set in config.py")

    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.EMBEDDINGS_DIR / "glove.6B.zip"

    print(f"Downloading GloVe from: {url}")
    print(f"Saving to: {zip_path}")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting embeddings...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        target_name = config.GLOVE_FILE.name
        if target_name not in zf.namelist():
            raise FileNotFoundError(
                f"Expected {target_name} inside {zip_path}, but it was not found. "
                f"Available: {zf.namelist()}"
            )
        zf.extract(target_name, path=config.EMBEDDINGS_DIR)

    if not config.GLOVE_FILE.exists():
        raise FileNotFoundError(f"Extraction completed but file still missing: {config.GLOVE_FILE}")


def main():
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    ensure_glove_file()

    texts, y = load_text_and_labels(
        tfds_dataset=config.TFDS_DATASET,
        hf_dataset=config.HF_DATASET,
        hf_split=config.HF_SPLIT,
        labels=config.LABELS,
    )

    max_examples = getattr(config, "MAX_TRAIN_EXAMPLES", None)
    if isinstance(max_examples, int) and max_examples > 0:
        texts = texts[:max_examples]
        y = y[:max_examples]

    prepared = prepare_tokenizer_and_sequences(
        texts=texts,
        y=y,
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_LEN,
        val_fraction=0.05,
        seed=config.RANDOM_SEED,
    )

    embedding_matrix = build_embedding_matrix(
        glove_path=config.GLOVE_FILE,
        word_index=prepared.tokenizer.word_index,
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
    )

    model = build_model(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        max_len=config.MAX_LEN,
        embedding_matrix=embedding_matrix,
        num_labels=len(config.LABELS),
        lstm_units=config.LSTM_UNITS,
        dense_units=config.DENSE_UNITS,
        dropout=config.DROPOUT,
        spatial_dropout=config.SPATIAL_DROPOUT,
        learning_rate=config.LEARNING_RATE,
    )

    # Optional 2-stage training: first keep embeddings frozen, then unfreeze for fine-tuning.
    finetune = bool(getattr(config, "FINETUNE_EMBEDDING", False))
    finetune_lr = float(getattr(config, "FINETUNE_LEARNING_RATE", 1e-4))
    finetune_epochs = int(getattr(config, "FINETUNE_EPOCHS", 2))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.ARTIFACTS_DIR / "model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    history1 = model.fit(
        prepared.x_train,
        prepared.y_train,
        validation_data=(prepared.x_val, prepared.y_val),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    history = history1

    if finetune and finetune_epochs > 0:
        emb_layer = model.get_layer("glove_embedding")
        emb_layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=len(config.LABELS), name="auc")],
        )

        history2 = model.fit(
            prepared.x_train,
            prepared.y_train,
            validation_data=(prepared.x_val, prepared.y_val),
            batch_size=config.BATCH_SIZE,
            epochs=finetune_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        merged = dict(history1.history)
        for k, v in history2.history.items():
            merged[k] = list(merged.get(k, [])) + list(v)

        class _H:
            def __init__(self, h):
                self.history = h

        history = _H(merged)

    tokenizer_json = prepared.tokenizer.to_json()
    (config.ARTIFACTS_DIR / "tokenizer.json").write_text(tokenizer_json, encoding="utf-8")
    (config.ARTIFACTS_DIR / "labels.json").write_text(json.dumps(config.LABELS), encoding="utf-8")

    (config.ARTIFACTS_DIR / "history.json").write_text(
        json.dumps(history.history, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
