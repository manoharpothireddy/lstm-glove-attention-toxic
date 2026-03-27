from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"

TFDS_DATASET = "wikipedia_toxicity_subtypes"
HF_DATASET = "thesofakillers/jigsaw-toxic-comment-classification-challenge"
HF_SPLIT = "train"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

RANDOM_SEED = 42

MAX_LEN = 128
VOCAB_SIZE = 30000

EMBEDDING_DIM = 100
GLOVE_FILE = EMBEDDINGS_DIR / "glove.6B.100d.txt"
GLOVE_ZIP_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
AUTO_DOWNLOAD_GLOVE = True

MAX_TRAIN_EXAMPLES = 50000

BATCH_SIZE = 256
EPOCHS = 2
LEARNING_RATE = 1e-3

LSTM_UNITS = 64
DENSE_UNITS = 128
DROPOUT = 0.3
SPATIAL_DROPOUT = 0.2
