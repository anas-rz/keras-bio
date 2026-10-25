# ProtT5

**ProtT5** (Rostlab) is a sequence-to-sequence foundation model trained on the BFD (Big Fantastic Database) and UniRef50 protein sequence databases using the T5 encoder-decoder architecture.

## Overview

- **Architecture**: T5 Transformer with relative attention bias buckets, RMS LayerNorm, and gated/ungated feed-forward networks.
- **Tasks**: Feature extraction (per-protein and per-residue embeddings), sequence classification, token classification, and conditional protein generation.
- **Module**: `keras_bio.protein.prot_t5`

---

## Quickstart

```python
from keras_bio.protein import (
    ProtT5Config,
    ProtT5EncoderModel,
    ProtT5Tokenizer,
)

# 1. Zero-dependency Tokenizer
tokenizer = ProtT5Tokenizer()
sequences = ["PRTEINO", "MKTIIALSYIFCLVFA"]

# 2. Load pretrained ProtT5-XL encoder weights from Hugging Face
model = ProtT5EncoderModel.from_hf("Rostlab/prot_t5_xl_half_uniref50-enc")

# 3. Extract Representations
# Per-protein pooled embedding (shape: [batch_size, hidden_dim])
pooled = model.embed(sequences, tokenizer=tokenizer, pool="mean")

# Per-residue embeddings (shape: [batch_size, max_length, hidden_dim])
residue_emb = model.embed(sequences, tokenizer=tokenizer, pool="none")
```

---

## Downstream Heads

The heads below use a small randomly initialized configuration so they run quickly;
pass `model.config` to build them on the full ProtT5-XL architecture.

### Sequence Classification

```python
from keras_bio.protein import ProtT5ForSequenceClassification

config = ProtT5Config(d_model=256, d_ff=1024, num_layers=4, num_heads=4)
batch = tokenizer(sequences, padding=True, return_tensors="np")

classifier = ProtT5ForSequenceClassification(config, num_classes=3)
logits = classifier(batch["input_ids"], attention_mask=batch["attention_mask"])
```

### Token Classification

```python
from keras_bio.protein import ProtT5ForTokenClassification

# Useful for secondary structure prediction or binding site detection
tagger = ProtT5ForTokenClassification(config, num_classes=8)
per_residue_logits = tagger(batch["input_ids"], attention_mask=batch["attention_mask"])
```

