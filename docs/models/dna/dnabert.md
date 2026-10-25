# DNABERT

**DNABERT** is a foundation model tailored for genomic DNA sequences that uses overlapping $k$-mer tokenization and bidirectional Transformer encoders.

## Overview

- **Tokenization**: Overlapping $k$-mers ($k=3, 4, 5, 6$) translating nucleotide strings (e.g. `ATCG...`) into $k$-mer tokens.
- **Tasks**: Promoter detection, transcription factor binding site (TFBS) identification, splice site prediction, and masked genomic language modeling.
- **Module**: `keras_bio.dna.dnabert`

---

## Quickstart

```python
from keras_bio.dna.dnabert import (
    DNABERTConfig,
    DNABERTModel,
    DNABERTTokenizer,
)

# 1. Initialize 6-mer genomic tokenizer
tokenizer = DNABERTTokenizer(kmer=6)
sequences = [
    "ATCGATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTA",
]

# 2. Tokenize DNA sequences
batch = tokenizer(sequences, padding=True, return_tensors="np")

# 3. Model instantiation
config = DNABERTConfig.from_preset("dnabert_6")
model = DNABERTModel(config)

# 4. Extract sequence representations
embeddings = model.embed(sequences, tokenizer=tokenizer, pool="mean")
```

---

## Classification Heads

```python
from keras_bio.dna.dnabert import DNABERTForSequenceClassification

classifier = DNABERTForSequenceClassification(config, num_labels=2)
logits = classifier(batch["input_ids"], attention_mask=batch["attention_mask"])
```

