# 3D Structure & PDB Export

`keras-bio` provides a protein structure pipeline to inspect and serialize predicted 3D atomic structures into standard PDB (Protein Data Bank) files.

## Coordinates & Atom Representations

Protein structures are represented using two complementary coordinate systems:

- **Atom14**: A compact representation where each residue contains up to 14 heavy atoms. Ideal for memory-efficient tensor transformations during the Structure Module iterations.
- **Atom37**: The full standard chemical representation containing all 37 distinct heavy atom types across natural amino acids.

---

## The ProteinStructure Object

```python
import numpy as np
from keras_bio.structure.esmfold import ProteinStructure, protein_to_pdb

# Construct a ProteinStructure data container
prot = ProteinStructure(
    aatype=np.array([0, 1, 2]),                  # Amino acid types (0=Ala, 1=Arg, 2=Asn)
    atom_positions=np.zeros((3, 37, 3)),         # 3D Cartesian coordinates in Angstroms
    atom_mask=np.ones((3, 37)),                  # Binary atom existence mask
    residue_index=np.array([1, 2, 3]),           # 1-based residue numbers
    b_factors=np.array([92.5, 88.0, 75.3]),      # pLDDT confidence scores (0..100)
    chain_tag="A",
)

# Convert to standard PDB text
pdb_string = protein_to_pdb(prot)
print(pdb_string)
```

---

## Visualizing Predicted Structures

You can open the generated `.pdb` files in any molecular viewer:

- **PyMOL**: `pymol predicted_structure.pdb`
- **ChimeraX**: `chimerax predicted_structure.pdb`
- **Mol\*** / **NGL Viewer**: In Jupyter notebooks or web browsers

