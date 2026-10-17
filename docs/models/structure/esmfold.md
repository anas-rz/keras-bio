# ESMFold

**ESMFold** (Meta AI) predicts full 3D atomic protein structures directly from primary amino acid sequence strings in seconds, bypassing the need for computationally heavy multiple sequence alignments (MSAs).

## Architecture & Components

1. **ESM-2 Language Model Backbone**: Computes evolutionary representations across all Transformer layers.
2. **Folding Trunk (Evoformer-like)**:
    - Triangular multiplicative updates (incoming and outgoing)
    - Triangular self-attention (starting and ending nodes)
    - Sequence-to-pair and pair-to-sequence bidirectional communication
3. **Structure Module**:
    - 3D Lie Group geometric primitives ($SO(3)$ rotations and $SE(3)$ rigid frames) in pure `keras.ops`
    - Invariant Point Attention (IPA)
    - Quaternion backbone updates & Angle ResNet for 7 torsion angles per residue
    - Atom14 and Atom37 coordinate reconstruction
4. **Confidence Heads**:
    - Residue-level pLDDT confidence (0–100 scale)
    - Predicted TM-score (pTM) and Predicted Aligned Error (PAE) matrix
5. **Module**: `keras_bio.structure.esmfold`

---

## Quickstart

```python
from keras_bio.structure.esmfold import (
    ESMFoldConfig,
    ESMFoldModel,
)

# 1. Initialize ESMFold
config = ESMFoldConfig.from_preset("esmfold_tiny")
model = ESMFoldModel(config)

# 2. Fold amino acid sequences directly
results = model.infer(["MLKNVQVQLV", "MKTIIALSYIFCLVFA"], num_recycles=1)

# 3. Access atomic coordinates & confidence scores
plddt = results["plddt"]  # [batch, seq_len, 37]
ptm = results["ptm"]      # [batch]
coords = results["atom37_positions"]  # [batch, seq_len, 37, 3]

# 4. Export standard PDB string
pdb_text = results["pdbs"][0]
with open("predicted_structure.pdb", "w") as f:
    f.write(pdb_text)
```

---

## Low-Level Tensor Execution

```python
import numpy as np
from keras import ops

# Input AF2 residue indices: Ala=0, Arg=1, Asn=2, Asp=3, ...
input_ids = ops.convert_to_tensor(np.array([[0, 1, 2, 3]], dtype="int32"))
attention_mask = ops.convert_to_tensor(np.ones((1, 4), dtype="float32"))

outputs = model(input_ids, attention_mask=attention_mask, num_recycles=0)

print("Distogram logits:", outputs["distogram_logits"].shape)  # [1, 4, 4, 64]
print("Atomic coordinates:", outputs["positions"].shape)       # [1, num_blocks, 4, 14, 3]
```

