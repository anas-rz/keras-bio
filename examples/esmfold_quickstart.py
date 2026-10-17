"""ESMFold Quickstart in Keras-Bio.

Demonstrates:
1. Initializing ESMFold structure prediction model
2. Direct 3D structure prediction from amino acid sequences (.infer)
3. Inspecting residue-level pLDDT confidence scores and predicted TM-score (pTM)
4. Exporting atomic coordinates to standard PDB (Protein Data Bank) files
5. Batched tensor-level forward pass with recycling
"""

import keras
from keras import ops
import numpy as np

from keras_bio.structure.esmfold import (
    ESMFoldConfig,
    ESMFoldModel,
    ProteinStructure,
    protein_to_pdb,
)

print(f"Running Keras with backend: {keras.backend.backend()}")

# 1. Initialize ESMFold model (using lightweight esmfold_tiny preset for rapid execution)
print("\n--- 1. Initializing ESMFold Model ---")
config = ESMFoldConfig.from_preset("esmfold_tiny")
model = ESMFoldModel(config)
print(f"Model initialized: trunk={config.trunk.num_blocks} blocks, sm={config.trunk.structure_module.num_blocks} blocks")

# 2. Sequence to 3D Structure Prediction (.infer)
print("\n--- 2. End-to-End Sequence Folding ---")
sequences = [
    "MLKNVQVQLV",
    "MKTIIALSYIFCLVFA",
]

result = model.infer(sequences, num_recycles=1)

plddt_scores = ops.convert_to_numpy(result["plddt"])
ptm_scores = ops.convert_to_numpy(result["ptm"])
atom37_coords = ops.convert_to_numpy(result["atom37_positions"])

print(f"Predicted atom37 coordinate tensor shape: {atom37_coords.shape}")  # (2, L, 37, 3)
print(f"Predicted pLDDT confidence tensor shape: {plddt_scores.shape}")     # (2, L, 37)
print(f"Predicted TM scores: {ptm_scores}")

for i, seq in enumerate(sequences):
    mean_plddt = np.mean(plddt_scores[i][: len(seq), 1])  # CA atom pLDDT
    print(f"Sequence {i} ({seq}): mean CA pLDDT = {mean_plddt:.2f}")

# 3. Export to Standard PDB Format
print("\n--- 3. PDB Export ---")
pdb_strings = result["pdbs"]
print(f"Generated {len(pdb_strings)} PDB string(s).")

# Display first few lines of the first predicted protein PDB
sample_lines = pdb_strings[0].strip().split("\n")
print(f"Sample PDB header for '{sequences[0]}':")
for line in sample_lines[:8]:
    print(" ", line)
print("  ...")
print(" ", sample_lines[-1])  # END

# 4. Low-Level Batched Forward Pass
print("\n--- 4. Batched Tensor Forward Pass ---")
# Amino acid integer indices (AF2 aatype: Ala=0, Arg=1, Asn=2, ...)
input_ids = ops.convert_to_tensor(np.array([[0, 1, 2, 3]], dtype="int32"))
attention_mask = ops.convert_to_tensor(np.ones((1, 4), dtype="float32"))

outputs = model(input_ids, attention_mask=attention_mask, num_recycles=0)
print("Output dictionary keys:", sorted(list(outputs.keys())))
print("Distogram logits shape:", ops.convert_to_numpy(outputs["distogram_logits"]).shape)
print("Backbone frames shape:", ops.convert_to_numpy(outputs["frames"]).shape)
print("\nESMFold demonstration completed successfully!")

