# Keras-Bio

Multi-backend biological sequence and structure foundation models implemented in **Keras 3**.

Runs seamlessly on **TensorFlow**, **PyTorch**, and **JAX**.

---

## Roadmap & Domains

- **Protein Sequences**: ProtT5 (Implemented), ESM (Next)
- **DNA Sequences**: DNABERT, HyenaDNA
- **Structures**: AlphaFold, ESMFold
- **Molecules/Drugs**: ChemBERTa, MolFormer
- **Single-Cell Omics**: Geneformer, scGPT

---

## Installation

```bash
pip install keras-bio
```

To run on a specific backend, set the `KERAS_BACKEND` environment variable:

```bash
export KERAS_BACKEND=jax        # or 'torch' or 'tensorflow'
```

---

## ProtT5 Quickstart

### 1. Tokenize & Extract Protein Embeddings

```python
from keras_bio.protein import ProtT5Config, ProtT5EncoderModel, ProtT5Tokenizer

# 1. Dependency-free protein sequence tokenizer
tokenizer = ProtT5Tokenizer()
sequences = ["PRTEINO", "MKTIIALSYIFCLVFA"]

# 2. Initialize model or load pretrained from Hugging Face Hub
model = ProtT5EncoderModel.from_hf("Rostlab/prot_t5_xl_half_uniref50-enc")

# 3. Extract pooled per-protein or per-residue embeddings
per_protein_emb = model.embed(sequences, tokenizer=tokenizer, pool="mean")  # (2, 1024)
per_residue_emb = model.embed(sequences, tokenizer=tokenizer, pool="none")  # (2, max_len, 1024)
```

### 2. Downstream Heads

```python
from keras_bio.protein import ProtT5ForSequenceClassification, ProtT5ForTokenClassification

# Sequence classification (e.g. cellular localization, enzyme function)
classifier = ProtT5ForSequenceClassification(model.config, num_classes=5)

# Per-residue token classification (e.g. secondary structure, binding sites)
token_classifier = ProtT5ForTokenClassification(model.config, num_classes=3)
```

---

## Architecture & Parity

- **Multi-backend**: Strict symbolic-safe operations (`keras.ops`) across TensorFlow, PyTorch, and JAX.
- **Feed-Forward**: Supports both ungated (ProtT5-XL) and gated (ProtT5-base-mt, T5-v1.1, mT5) projections.
- **Parity Checked**: Validated against PyTorch Hugging Face reference implementations with numerical difference $< 10^{-6}$.
- **Serialization**: Native `.keras` zip-format serialization (`model.save()` / `keras.saving.load_model()`).

---

## Running Tests

```bash
# Run layer and model tests
pytest tests/ -v

# Run across backends
KERAS_BACKEND=tensorflow pytest tests/protein/test_prot_t5_multibackend.py -v
KERAS_BACKEND=torch pytest tests/protein/test_prot_t5_multibackend.py -v
KERAS_BACKEND=jax pytest tests/protein/test_prot_t5_multibackend.py -v
```

