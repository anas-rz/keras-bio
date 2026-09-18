"""ProtT5 Quickstart in Keras-Bio.

Demonstrates:
1. Dependency-free protein sequence tokenization
2. Residue and protein-level embedding extraction
3. Downstream classification heads
"""

import keras
import numpy as np

from keras_bio.protein import (
    ProtT5Config,
    ProtT5EncoderModel,
    ProtT5ForSequenceClassification,
    ProtT5Tokenizer,
)

print(f"Running Keras with backend: {keras.backend.backend()}")

# 1. Initialize tokenizer and sample protein sequences
tokenizer = ProtT5Tokenizer()

sequences = [
    "PRTEINO",  # contains non-standard 'O' -> cleaned to 'X'
    "SEQWENCE",
    "MKTIIALSYIFCLVFA",  # signal peptide
]

print("\n--- 1. Tokenization ---")
batch = tokenizer(sequences, padding=True, return_tensors="np")
print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
for i, seq in enumerate(sequences):
    print(f"Seq {i}: {seq} -> Decoded: {tokenizer.decode(batch['input_ids'][i])}")

# 2. Initialize ProtT5 model (using a compact config for fast local demo)
config = ProtT5Config(
    vocab_size=128,
    d_model=128,
    d_kv=32,
    d_ff=512,
    num_layers=4,
    num_heads=4,
    dropout_rate=0.0,
)
encoder = ProtT5EncoderModel(config)

print("\n--- 2. Protein Embedding Extraction ---")
# Direct high-level sequence string embedding:
per_protein_emb = encoder.embed(sequences, tokenizer=tokenizer, pool="mean")
print("Per-protein pooled embeddings shape:", per_protein_emb.shape)  # (3, 128)

per_residue_emb = encoder.embed(sequences, tokenizer=tokenizer, pool="none")
print("Per-residue embeddings shape:", per_residue_emb.shape)  # (3, max_len, 128)

# 3. Protein Sequence Classification
print("\n--- 3. Sequence Classification Head ---")
classifier = ProtT5ForSequenceClassification(config, num_classes=3)
cls_out = classifier(batch)
print("Classification logits shape:", cls_out["logits"].shape)  # (3, 3)
probs = keras.ops.softmax(cls_out["logits"], axis=-1)
print("Predicted probabilities for seq 0:", keras.ops.convert_to_numpy(probs[0]))

print("\nProtT5 Keras-Bio Quickstart successfully finished!")

