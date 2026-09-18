"""DNABERT Quickstart in Keras-Bio.

Demonstrates:
1. Zero-dependency k-mer DNA tokenization (continuous sequences and space-separated k-mers)
2. Sequence-level and per-kmer genomic embedding extraction
3. Zero-shot masked k-mer prediction (Masked LM)
4. Sequence classification head (e.g., promoter vs. non-promoter)
5. Token classification head (e.g., transcription factor binding site detection)
"""

import keras
from keras import ops
import numpy as np

from keras_bio.dna import (
    DNABERTConfig,
    DNABERTForMaskedLM,
    DNABERTForSequenceClassification,
    DNABERTForTokenClassification,
    DNABERTModel,
    DNABERTTokenizer,
)

print(f"Running Keras with backend: {keras.backend.backend()}")

# 1. Initialize 6-mer tokenizer and sample DNA sequences
tokenizer = DNABERTTokenizer(kmer=6)

dna_sequences = [
    "ATCGATCGATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTA",
]

print("\n--- 1. Tokenization ---")
batch = tokenizer(dna_sequences, padding=True, return_tensors="np")
print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
for i, seq in enumerate(dna_sequences):
    kmers = tokenizer.dna_to_kmers(seq)
    print(f"Seq {i} ({len(seq)} bp): {len(kmers)} 6-mers -> Decoded: {tokenizer.decode(batch['input_ids'][i])[:30]}...")

# 2. Compact DNABERT model for fast local demonstration
config = DNABERTConfig(
    kmer=6,
    vocab_size=tokenizer.vocab_size,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
)
dna_model = DNABERTModel(config)

print("\n--- 2. Genomic Embedding Extraction ---")
# High-level sequence embedding with mean pooling over k-mers
pooled_emb = dna_model.embed(dna_sequences, tokenizer=tokenizer, pool="mean")
print("Per-sequence pooled embeddings shape (mean):", pooled_emb.shape)  # (2, 128)

# [CLS] representation pooling
cls_emb = dna_model.embed(dna_sequences, tokenizer=tokenizer, pool="cls")
print("Per-sequence pooled embeddings shape (cls):", cls_emb.shape)  # (2, 128)

# Per-kmer embeddings
per_kmer_emb = dna_model.embed(dna_sequences, tokenizer=tokenizer, pool=None)
print("Per-kmer embeddings shape:", per_kmer_emb.shape)  # (2, max_len, 128)

# 3. Masked Language Modeling
print("\n--- 3. Masked Language Modeling ---")
masked_lm = DNABERTForMaskedLM(config)
encoded_masked = tokenizer(dna_sequences[0], return_tensors="np")
# Mask out 2nd k-mer (index 2, right after [CLS] at 0 and first k-mer at 1)
masked_pos = 2
original_id = encoded_masked["input_ids"][0, masked_pos]
encoded_masked["input_ids"][0, masked_pos] = tokenizer.mask_token_id

lm_out = masked_lm(
    {
        "input_ids": ops.convert_to_tensor(encoded_masked["input_ids"]),
        "attention_mask": ops.convert_to_tensor(encoded_masked["attention_mask"]),
    }
)
logits = lm_out["logits"]
pred_id = ops.convert_to_numpy(ops.argmax(logits[0, masked_pos], axis=-1))
print(f"Masked position {masked_pos} original token:", tokenizer.id_to_token[int(original_id)])
print(f"Masked position {masked_pos} predicted token:", tokenizer.id_to_token[int(pred_id)])

# 4. Sequence Classification Head (e.g. Promoter vs Non-promoter)
print("\n--- 4. Sequence Classification Head ---")
input_ids_tensor = ops.convert_to_tensor(batch["input_ids"])
mask_tensor = ops.convert_to_tensor(batch["attention_mask"])

classifier = DNABERTForSequenceClassification(config, num_labels=2)
cls_out = classifier(
    {
        "input_ids": input_ids_tensor,
        "attention_mask": mask_tensor,
    }
)
probs = ops.softmax(cls_out["logits"], axis=-1)
print("Predicted sequence probabilities for seq 0:", ops.convert_to_numpy(probs[0]))

# 5. Token Classification Head (e.g. TF binding site across k-mers)
print("\n--- 5. Token Classification Head ---")
token_classifier = DNABERTForTokenClassification(config, num_labels=4)
token_out = token_classifier(
    {
        "input_ids": input_ids_tensor,
        "attention_mask": mask_tensor,
    }
)
token_logits = token_out["logits"]
print("Token classification logits shape:", token_logits.shape)

print("\nDNABERT Keras-Bio Quickstart successfully finished!")

