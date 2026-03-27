from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_best"
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

MAX_LEN = 220
VOCAB_SIZE = 80000

EMBEDDING_DIM = 300
GLOVE_FILE = EMBEDDINGS_DIR / "glove.6B.300d.txt"
GLOVE_ZIP_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
AUTO_DOWNLOAD_GLOVE = True

# Use full dataset (set to 0 / None to disable cap)
MAX_TRAIN_EXAMPLES = 0

BATCH_SIZE = 128
EPOCHS = 6
LEARNING_RATE = 1e-3

# Fine-tune embeddings after initial training
FINETUNE_EMBEDDING = True
FINETUNE_EPOCHS = 3
FINETUNE_LEARNING_RATE = 1e-4

LSTM_UNITS = 192
DENSE_UNITS = 256
DROPOUT = 0.4
SPATIAL_DROPOUT = 0.2
