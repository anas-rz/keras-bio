# ESM-2 (Evolutionary Scale Modeling)

**ESM-2** (Meta AI) is a state-of-the-art protein language model trained across millions of natural evolutionary protein sequences using masked language modeling.

## Overview

- **Architecture**: Bidirectional Transformer Encoder with Rotary Position Embeddings (RoPE) and learned residue-residue contact prediction heads using Average Product Correction (APC).
- **Available Presets**:
  - `esm2_t6_8M` (6 layers, 320 dim)
  - `esm2_t12_35M` (12 layers, 480 dim)
  - `esm2_t30_150M` (30 layers, 640 dim)
  - `esm2_t33_650M` (33 layers, 1280 dim)
  - `esm2_t36_3B` (36 layers, 2560 dim)
  - `esm2_t48_15B` (48 layers, 5120 dim)
- **Module**: `keras_bio.protein.esm2`

---

## Quickstart

```python
from keras_bio.protein.esm2 import (
    ESM2Config,
    ESM2Model,
    ESMTokenizer,
)

tokenizer = ESMTokenizer()
sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

config = ESM2Config.from_preset("esm2_t6_8M")
model = ESM2Model(config)

# High-level embedding extraction
embeddings = model.embed(sequences, tokenizer=tokenizer, pool="mean")
```

---

## Contact Prediction

ESM-2 can predict residue-residue contacts directly from attention maps using the contact prediction head with Average Product Correction (APC):

```python
batch = tokenizer(sequences, return_tensors="np")
contact_map = model.predict_contacts(batch["input_ids"], attention_mask=batch["attention_mask"])
print("Predicted contact map shape:", contact_map.shape)  # [batch, seq_len, seq_len]
```

---

## Masked Language Modeling

```python
from keras_bio.protein.esm2 import ESM2ForMaskedLM

mlm_model = ESM2ForMaskedLM(config)
logits = mlm_model(batch["input_ids"])["logits"]
```

