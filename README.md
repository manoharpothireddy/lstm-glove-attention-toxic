# LSTM-GloVe Fusion with Attention (Multi-Label Toxic Comment)

## Dataset
This project uses TFDS `wikipedia_toxicity_subtypes` (Jigsaw 6-label toxicity).

## Setup
1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Download and extract GloVe 6B
- Download: https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
- Extract `glove.6B.300d.txt` into `./embeddings/`

3. Train

```bash
python src/train.py
```

4. Evaluate

```bash
python src/evaluate.py
```

5. Inference

```bash
python src/inference.py
```
