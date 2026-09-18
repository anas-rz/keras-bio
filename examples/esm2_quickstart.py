"""ESM-2 Quickstart in Keras-Bio.

Demonstrates:
1. Zero-dependency ESM-2 protein sequence tokenization
2. Residue and protein-level embedding extraction
3. Residue-residue contact map prediction via attention APC
4. Zero-shot masked residue prediction (Masked LM)
5. Sequence classification head
"""

import keras
from keras import ops
import numpy as np

from keras_bio.protein.esm2 import (
    ESM2Config,
    ESM2ForMaskedLM,
    ESM2ForSequenceClassification,
    ESM2Model,
    ESMTokenizer,
)

print(f"Running Keras with backend: {keras.backend.backend()}")

# 1. Initialize tokenizer and sample protein sequences
tokenizer = ESMTokenizer()

sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "MKTIIALSYIFCLVFA",
]

print("\n--- 1. Tokenization ---")
batch = tokenizer(sequences, padding=True, return_tensors="np")
print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
for i, seq in enumerate(sequences):
    print(f"Seq {i}: {seq[:10]}... -> Decoded: {tokenizer.decode(batch['input_ids'][i])[:15]}...")

# 2. Compact ESM-2 model for fast local demonstration
config = ESM2Config(
    vocab_size=33,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
)
esm_model = ESM2Model(config)

print("\n--- 2. Protein Embedding Extraction ---")
# High-level sequence embedding with mean pooling across residues
per_protein_emb = esm_model.embed(sequences, tokenizer=tokenizer, pool="mean")
print("Per-protein pooled embeddings shape:", per_protein_emb.shape)  # (2, 128)

# Per-residue embeddings
per_residue_emb = esm_model.embed(sequences, tokenizer=tokenizer, pool="none")
print("Per-residue embeddings shape:", per_residue_emb.shape)  # (2, max_len, 128)

# 3. Residue-residue Contact Prediction
print("\n--- 3. Contact Map Prediction ---")
input_ids_tensor = ops.convert_to_tensor(batch["input_ids"])
mask_tensor = ops.convert_to_tensor(batch["attention_mask"])
contacts = esm_model.predict_contacts(input_ids_tensor, attention_mask=mask_tensor)
print("Predicted contact maps shape:", contacts.shape)
# Contact probabilities are in [0, 1]
contacts_np = ops.convert_to_numpy(contacts)
print(f"Contact map min: {contacts_np.min():.4f}, max: {contacts_np.max():.4f}")

# 4. Masked Language Modeling
print("\n--- 4. Masked Language Modeling ---")
masked_lm = ESM2ForMaskedLM(config)
masked_seq = "M<mask >TV"  # or with mask token id
encoded_masked = tokenizer("MKTV", return_tensors="np")
# Replace second residue (index 2) with <mask > token (id 32)
encoded_masked["input_ids"][0, 2] = tokenizer.mask_token_id
lm_out = masked_lm(
    {
        "input_ids": ops.convert_to_tensor(encoded_masked["input_ids"]),
        "attention_mask": ops.convert_to_tensor(encoded_masked["attention_mask"]),
    }
)
logits = lm_out["logits"]
pred_id = ops.convert_to_numpy(ops.argmax(logits[0, 2], axis=-1))
print("Masked position predicted token:", tokenizer.id_to_token[int(pred_id)])

# 5. Sequence Classification Head
print("\n--- 5. Sequence Classification Head ---")
classifier = ESM2ForSequenceClassification(config, num_labels=3)
cls_out = classifier(
    {
        "input_ids": input_ids_tensor,
        "attention_mask": mask_tensor,
    }
)
probs = ops.softmax(cls_out["logits"], axis=-1)
print("Predicted probabilities for seq 0:", ops.convert_to_numpy(probs[0]))

print("\nESM-2 Keras-Bio Quickstart successfully finished!")

