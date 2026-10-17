"""
PDB structure generation utilities for predicted 3D protein atomic coordinates.

Translates predicted atom37 coordinates, pLDDT B-factors, and amino acid sequences
into valid standard PDB (Protein Data Bank) text files.
"""

from dataclasses import dataclass
import numpy as np

import keras_bio.structure.esmfold.residue_constants as rc


@dataclass
class ProteinStructure:
    """Represents a predicted 3D protein structure."""
    aatype: np.ndarray             # [N] integer residue indices (0..20)
    atom_positions: np.ndarray     # [N, 37, 3] Cartesian coordinates in Angstroms
    atom_mask: np.ndarray          # [N, 37] binary existence mask
    residue_index: np.ndarray      # [N] 1-based residue numbers
    b_factors: np.ndarray          # [N, 37] per-atom pLDDT confidence scores (0..100)
    chain_tag: str = "A"


def protein_to_pdb(prot: ProteinStructure) -> str:
    """
    Converts a ProteinStructure object into a standard PDB formatted string.

    Args:
        prot: ProteinStructure instance with atomic coordinates and metadata.

    Returns:
        PDB file formatted string.
    """
    restypes = rc.restypes + ["X"]

    def res_1to3(r: int) -> str:
        if 0 <= r < len(restypes):
            return rc.restype_1to3.get(restypes[r], "UNK")
        return "UNK"

    atom_types = rc.atom_types
    pdb_lines: list[str] = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    if b_factors.ndim == 1:
        b_factors = np.broadcast_to(b_factors[:, None], atom_mask.shape)
    if np.max(b_factors) <= 1.0:
        b_factors = b_factors * 100.0
    chain_tag = prot.chain_tag

    n = aatype.shape[0]
    atom_index = 1

    for i in range(n):
        res_name_3 = res_1to3(int(aatype[i]))
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]
            charge = ""

            # Standard 80-column fixed-width PDB ATOM record
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Add chain termination
    chain_end = "TER"
    chain_term_line = (
        f"{chain_end:<6}{atom_index:>5}      {res_1to3(int(aatype[-1])):>3} {chain_tag:>1}{residue_index[-1]:>4}"
    )
    pdb_lines.append(chain_term_line)
    pdb_lines.append("END")

    return "\n".join(pdb_lines) + "\n"

