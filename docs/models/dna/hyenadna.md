# HyenaDNA

**HyenaDNA** is a genomic foundation model that processes ultra-long DNA sequences at single-nucleotide resolution using subquadratic long convolutions via FFT.

## Overview

- **Tokenization**: Pure character-level single-nucleotide resolution (`A`, `C`, `G`, `T`, `N`).
- **Core Mechanism**: Hyena recurrence with implicit long convolution filters parametrized by MLPs, sine activations, and exponential window modulations. Convolutions are evaluated in $O(L \log L)$ time via the Convolution Theorem and Fast Fourier Transforms (`ops.rfft` / `ops.irfft`).
- **Context Length**: Scales from 1k up to 1M base pairs.
- **Module**: `keras_bio.dna.hyena`

---

## Quickstart

```python
from keras_bio.dna.hyena import (
    HyenaDNAConfig,
    HyenaDNAModel,
    HyenaDNATokenizer,
)

# 1. Single-nucleotide tokenizer
tokenizer = HyenaDNATokenizer()
dna_seqs = ["ACTG" * 32, "GATC" * 16]

# 2. Initialize Model
config = HyenaDNAConfig.from_preset("hyenadna_tiny_1k")
model = HyenaDNAModel(config)

# 3. Extract embeddings
embeddings = model.embed(dna_seqs, tokenizer=tokenizer, pool="mean")

# 4. Tokenize for direct model calls
batch = tokenizer(dna_seqs, padding=True, return_tensors="np")
```

---

## Causal Generation & Classification

```python
from keras_bio.dna.hyena import (
    HyenaDNAForCausalLM,
    HyenaDNAForSequenceClassification,
)

# Causal next-token prediction
lm_model = HyenaDNAForCausalLM(config)
logits = lm_model(batch["input_ids"])["logits"]

# Chromatin profile or enhancer classification
classifier = HyenaDNAForSequenceClassification(config, num_labels=2)
preds = classifier(batch["input_ids"])["logits"]
```

