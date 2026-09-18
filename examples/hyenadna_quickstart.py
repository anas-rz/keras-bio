"""HyenaDNA Quickstart in Keras-Bio.

Demonstrates:
1. Zero-dependency character-level nucleotide tokenization
2. Sequence-level and per-nucleotide genomic embedding extraction
3. Causal language modeling (next-nucleotide generation)
4. Sequence classification head (e.g., chromatin accessibility / promoter detection)
"""

import keras
from keras import ops
import numpy as np

from keras_bio.dna import (
    HyenaDNAConfig,
    HyenaDNAForCausalLM,
    HyenaDNAForSequenceClassification,
    HyenaDNAModel,
    HyenaDNATokenizer,
)

print(f"Running Keras with backend: {keras.backend.backend()}")

# 1. Initialize tokenizer and sample DNA sequences
tokenizer = HyenaDNATokenizer()

dna_sequences = [
    "ATCGATCGATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTA",
]

print("\n--- 1. Tokenization ---")
batch = tokenizer(dna_sequences, padding=True, return_tensors="np")
print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
for i, seq in enumerate(dna_sequences):
    print(f"Seq {i}: {seq[:15]}... -> Decoded: {tokenizer.decode(batch['input_ids'][i])[:15]}...")

# 2. Compact HyenaDNA model for fast demonstration
config = HyenaDNAConfig(
    d_model=128,
    n_layer=2,
    d_inner=512,
    vocab_size=12,
    max_seq_len=1026,
    filter_order=64,
    short_filter_order=3,
    emb_dim=5,
)
hyena_model = HyenaDNAModel(config)

print("\n--- 2. Genomic Embedding Extraction ---")
# Mean pooling across nucleotides
mean_emb = hyena_model.embed(dna_sequences, tokenizer=tokenizer, pool="mean")
print("Per-sequence pooled embeddings shape (mean):", mean_emb.shape)  # (2, 128)

# Last token pooling (causal representation)
last_emb = hyena_model.embed(dna_sequences, tokenizer=tokenizer, pool="last")
print("Per-sequence pooled embeddings shape (last):", last_emb.shape)  # (2, 128)

# Per-nucleotide embeddings
per_nt_emb = hyena_model.embed(dna_sequences, tokenizer=tokenizer, pool=None)
print("Per-nucleotide embeddings shape:", per_nt_emb.shape)  # (2, seq_len, 128)

# 3. Causal Language Modeling (Next-Nucleotide Prediction)
print("\n--- 3. Causal Language Modeling ---")
causal_lm = HyenaDNAForCausalLM(config)
input_ids_tensor = ops.convert_to_tensor(batch["input_ids"])
lm_outputs = causal_lm(input_ids_tensor)
logits = lm_outputs["logits"]  # (batch, seq_len, vocab_size)
print("Causal LM logits shape:", logits.shape)

# Predict next nucleotide for the first sequence
next_token_id = ops.convert_to_numpy(ops.argmax(logits[0, -2], axis=-1))
print("Next nucleotide predicted token:", tokenizer.id_to_token.get(int(next_token_id), "[UNK]"))

# 4. Sequence Classification Head
print("\n--- 4. Sequence Classification Head ---")
classifier = HyenaDNAForSequenceClassification(config, num_labels=2)
cls_out = classifier(input_ids_tensor)
probs = ops.softmax(cls_out["logits"], axis=-1)
print("Classification probabilities for seq 0:", ops.convert_to_numpy(probs[0]))

print("\nHyenaDNA Keras-Bio Quickstart successfully finished!")

