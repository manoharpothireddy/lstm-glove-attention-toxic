from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from datasets import load_dataset  # type: ignore
    from datasets import DownloadConfig  # type: ignore
    from datasets.exceptions import DatasetGenerationError  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None
    DownloadConfig = None
    DatasetGenerationError = None

try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # pragma: no cover
    hf_hub_download = None


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


def _as_np(ds):
    return tfds.as_numpy(ds)


@dataclass(frozen=True)
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    tokenizer: tf.keras.preprocessing.text.Tokenizer


def load_tfds_text_and_labels(dataset_name: str, labels: list[str]):
    ds = tfds.load(dataset_name, split="train", as_supervised=False)

    texts = []
    ys = []
    for ex in _as_np(ds):
        text = ex.get("text")
        if text is None:
            text = ex.get("comment_text")
        if text is None:
            raise KeyError("TFDS example has no 'text' or 'comment_text' field")

        text = text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else str(text)

        y_row = []
        for lbl in labels:
            v = ex.get(lbl)
            if v is None:
                raise KeyError(f"TFDS example missing label field: {lbl}")
            y_row.append(int(v))

        texts.append(text)
        ys.append(y_row)

    return texts, np.asarray(ys, dtype=np.float32)


def load_hf_text_and_labels(dataset_name: str, split: str, labels: list[str]):
    if load_dataset is None:
        raise ImportError(
            "Hugging Face 'datasets' is not installed. Install it with: pip install datasets"
        )

    download_config = None
    if DownloadConfig is not None:
        download_config = DownloadConfig(
            max_retries=20,
        )

    try:
        ds = load_dataset(dataset_name, split=split, download_config=download_config)
    except Exception as e:
        # On some Windows setups / flaky networks, datasets' Arrow cache build can fail with
        # missing *.arrow inside an '.incomplete' directory. Fall back to downloading the
        # raw CSV directly from the dataset repo and loading via pandas.
        print(f"HF datasets load failed ({type(e).__name__}: {e}). Falling back to direct CSV download...")
        return load_hf_csv_text_and_labels(dataset_name, split, labels)

    text_col = None
    for candidate in ("comment_text", "text", "comment"):
        if candidate in ds.column_names:
            text_col = candidate
            break
    if text_col is None:
        raise KeyError(f"No text column found in {dataset_name}. Columns: {ds.column_names}")

    for lbl in labels:
        if lbl not in ds.column_names:
            raise KeyError(f"Missing label '{lbl}' in {dataset_name}. Columns: {ds.column_names}")

    texts = [str(t) for t in ds[text_col]]
    y = np.vstack([np.asarray(ds[lbl], dtype=np.float32) for lbl in labels]).T
    return texts, y


def load_hf_csv_text_and_labels(dataset_name: str, split: str, labels: list[str]):
    if hf_hub_download is None:
        raise ImportError(
            "huggingface_hub is not installed but is required for the CSV fallback. "
            "Install it with: pip install huggingface_hub"
        )

    # Common conventions in Kaggle-like mirrors.
    candidate_files = []
    if split == "train":
        candidate_files = ["train.csv", "Train.csv"]
    elif split == "test":
        candidate_files = ["test.csv", "Test.csv"]
    else:
        candidate_files = [f"{split}.csv"]

    last_err = None
    csv_path = None
    for fname in candidate_files:
        try:
            csv_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=fname)
            break
        except Exception as e:
            last_err = e

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not download CSV from HF dataset repo '{dataset_name}'. Tried: {candidate_files}. "
            f"Last error: {last_err}"
        )

    df = pd.read_csv(csv_path)

    text_col = None
    for candidate in ("comment_text", "text", "comment"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise KeyError(f"No text column found in {dataset_name} CSV. Columns: {list(df.columns)}")

    for lbl in labels:
        if lbl not in df.columns:
            raise KeyError(f"Missing label '{lbl}' in {dataset_name} CSV. Columns: {list(df.columns)}")

    texts = df[text_col].astype(str).tolist()
    y = df[labels].astype(np.float32).values
    return texts, y


def load_text_and_labels(
    *,
    tfds_dataset: str,
    hf_dataset: str,
    hf_split: str,
    labels: list[str],
):
    try:
        return load_tfds_text_and_labels(tfds_dataset, labels)
    except Exception as e:
        print(f"TFDS load failed ({type(e).__name__}: {e}). Falling back to Hugging Face datasets...")
        return load_hf_text_and_labels(hf_dataset, hf_split, labels)


def prepare_tokenizer_and_sequences(
    *,
    texts: list[str],
    y: np.ndarray,
    vocab_size: int,
    max_len: int,
    val_fraction: float,
    seed: int,
) -> PreparedData:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)

    n_val = int(len(texts) * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(train_texts)

    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_len, padding="post", truncating="post"
    )
    x_val = tf.keras.preprocessing.sequence.pad_sequences(
        x_val, maxlen=max_len, padding="post", truncating="post"
    )

    return PreparedData(
        x_train=np.asarray(x_train, dtype=np.int32),
        y_train=y_train,
        x_val=np.asarray(x_val, dtype=np.int32),
        y_val=y_val,
        tokenizer=tokenizer,
    )


def build_embedding_matrix(
    *,
    glove_path,
    word_index: Dict[str, int],
    vocab_size: int,
    embedding_dim: int,
):
    embeddings_index: Dict[str, np.ndarray] = {}

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")
            if len(values) < embedding_dim + 1:
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            if coefs.shape[0] != embedding_dim:
                continue
            embeddings_index[word] = coefs

    matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim)).astype(np.float32)
    matrix[0] = 0.0

    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            matrix[i] = vec

    return matrix
