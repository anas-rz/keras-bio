"""
Feature transformation utilities for protein structure prediction.

Translates predicted torsion angles to 3D reference frames, maps residue
frames to atom14 atomic positions, and expands atom14 to atom37 coordinates.
"""

from typing import Any

import keras
from keras import ops

import keras_bio.structure.esmfold.geometry as geometry
import keras_bio.structure.esmfold.residue_constants as rc


def torsion_angles_to_frames(
    r: geometry.Rigid,
    alpha,
    aatype,
    rrgdf,
) -> geometry.Rigid:
    """
    Converts predicted torsion angles to global 3D reference frames for each residue.

    Args:
        r: [*, N] Rigid backbone-to-global transformation.
        alpha: [*, N, 7, 2] Predicted (sin, cos) torsion angles (omega, phi, psi, chi1, chi2, chi3, chi4).
        aatype: [*, N] Amino acid residue indices (0..20).
        rrgdf: [21, 8, 4, 4] Restype rigid group default frames tensor.

    Returns:
        [*, N, 8] Rigid transformations from local residue frames to global space.
    """
    # [*, N, 8, 4, 4]
    default_4x4 = ops.take(rrgdf, aatype, axis=0)
    default_r = geometry.Rigid.from_tensor_4x4(default_4x4)

    # Prepend identity rotation [0, 1] (sin=0, cos=1) for backbone group
    bb_rot_zeros = ops.zeros(ops.shape(alpha)[:-2] + (1, 1), dtype=alpha.dtype)
    bb_rot_ones = ops.ones(ops.shape(alpha)[:-2] + (1, 1), dtype=alpha.dtype)
    bb_rot = ops.concatenate([bb_rot_zeros, bb_rot_ones], axis=-1)  # [*, N, 1, 2]

    # [*, N, 8, 2]
    alpha = ops.concatenate([bb_rot, alpha], axis=-2)

    # Produces rotation matrices of the form:
    # [
    #   [1,    0,     0  ],
    #   [0,  cos,  -sin  ],
    #   [0,  sin,   cos  ]
    # ]
    sin_a = alpha[..., 0]
    cos_a = alpha[..., 1]
    ones = ops.ones(ops.shape(sin_a), dtype=alpha.dtype)
    zeros = ops.zeros(ops.shape(sin_a), dtype=alpha.dtype)

    row0 = ops.stack([ones, zeros, zeros], axis=-1)
    row1 = ops.stack([zeros, cos_a, -sin_a], axis=-1)
    row2 = ops.stack([zeros, sin_a, cos_a], axis=-1)
    all_rots = ops.stack([row0, row1, row2], axis=-2)  # [*, N, 8, 3, 3]

    all_frames = default_r.compose(geometry.Rigid(geometry.Rotation(rot_mats=all_rots), ops.zeros(ops.shape(all_rots)[:-1], dtype=all_rots.dtype)))

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = geometry.Rigid.cat(
        [
            all_frames[..., :5],
            geometry.Rigid(chi2_frame_to_bb.get_rots(), chi2_frame_to_bb.get_trans())[..., None],
            geometry.Rigid(chi3_frame_to_bb.get_rots(), chi3_frame_to_bb.get_trans())[..., None],
            geometry.Rigid(chi4_frame_to_bb.get_rots(), chi4_frame_to_bb.get_trans())[..., None],
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: geometry.Rigid,
    aatype,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    """
    Transforms residue rigid frames and ideal chemical literature positions into atom14 3D Cartesian coordinates.

    Args:
        r: [*, N, 8] Transformation objects for 8 frames per residue.
        aatype: [*, N] Amino acid residue types.
        default_frames: [21, 8, 4, 4] Default frame constants.
        group_idx: [21, 14] Mapping from atom14 index to rigid group index.
        atom_mask: [21, 14] Mask of valid atoms in atom14 representation.
        lit_positions: [21, 14, 3] Literature coordinates of atoms in local frames.

    Returns:
        [*, N, 14, 3] Global 3D atomic coordinates.
    """
    # [*, N, 14]
    group_mask = ops.take(group_idx, aatype, axis=0)

    # [*, N, 14, 8]
    num_classes = ops.shape(default_frames)[-3]
    group_mask_one_hot = ops.one_hot(group_mask, num_classes=num_classes)
    group_mask_one_hot = ops.cast(group_mask_one_hot, dtype=r.get_rots().get_rot_mats().dtype)

    # [*, N, 1, 8] -> [*, N, 14, 8]
    r_expanded = r[..., None, :]
    t_atoms_to_global = r_expanded * group_mask_one_hot

    # Map tensor fn: sum out the rigid group dimension (axis=-1 on trans, axis=-3 on rot_mats)
    def sum_groups(x):
        return ops.sum(x, axis=-2 if len(ops.shape(x)) > len(ops.shape(group_mask_one_hot)) else -1)

    # Alternatively, direct component summation for rot_mats [..., N, 14, 8, 3, 3] and trans [..., N, 14, 8, 3]
    summed_rots = ops.sum(t_atoms_to_global.get_rots().get_rot_mats(), axis=-3)
    summed_trans = ops.sum(t_atoms_to_global.get_trans(), axis=-2)
    t_atoms_to_global = geometry.Rigid(geometry.Rotation(rot_mats=summed_rots), summed_trans)

    # [*, N, 14, 1]
    res_atom_mask = ops.expand_dims(ops.take(atom_mask, aatype, axis=0), axis=-1)

    # [*, N, 14, 3]
    lit_pos = ops.take(lit_positions, aatype, axis=0)
    pred_positions = t_atoms_to_global.apply(lit_pos)
    pred_positions = pred_positions * res_atom_mask

    return pred_positions


def make_atom14_masks(aatype):
    """
    Constructs mapping tensors and existence masks for atom14 and atom37 representations.

    Args:
        aatype: [*, N] integer tensor of residue types (0..20).

    Returns:
        dict containing:
          - atom14_atom_exists: [*, N, 14]
          - atom37_atom_exists: [*, N, 37]
          - residx_atom14_to_atom37: [*, N, 14]
          - residx_atom37_to_atom14: [*, N, 37]
    """
    restype_atom14_to_atom37_list = []
    restype_atom37_to_atom14_list = []
    restype_atom14_mask_list = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37_list.append([(rc.atom_order[name] if name else 0) for name in atom_names])
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14_list.append([(atom_name_to_idx14.get(name, 0)) for name in rc.atom_types])
        restype_atom14_mask_list.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK' / unknown
    restype_atom14_to_atom37_list.append([0] * 14)
    restype_atom37_to_atom14_list.append([0] * 37)
    restype_atom14_mask_list.append([0.0] * 14)

    restype_atom14_to_atom37 = ops.convert_to_tensor(restype_atom14_to_atom37_list, dtype="int32")
    restype_atom37_to_atom14 = ops.convert_to_tensor(restype_atom37_to_atom14_list, dtype="int32")
    restype_atom14_mask = ops.convert_to_tensor(restype_atom14_mask_list, dtype="float32")

    residx_atom14_to_atom37 = ops.take(restype_atom14_to_atom37, aatype, axis=0)
    residx_atom37_to_atom14 = ops.take(restype_atom37_to_atom14, aatype, axis=0)
    atom14_atom_exists = ops.take(restype_atom14_mask, aatype, axis=0)

    # Restype atom37 mask
    import numpy as np
    restype_atom37_mask_np = np.zeros([21, 37], dtype=np.float32)
    for restype_idx, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask_np[restype_idx, atom_type] = 1.0

    restype_atom37_mask = ops.convert_to_tensor(restype_atom37_mask_np, dtype="float32")
    atom37_atom_exists = ops.take(restype_atom37_mask, aatype, axis=0)

    return {
        "atom14_atom_exists": atom14_atom_exists,
        "atom37_atom_exists": atom37_atom_exists,
        "residx_atom14_to_atom37": residx_atom14_to_atom37,
        "residx_atom37_to_atom14": residx_atom37_to_atom14,
    }


def atom14_to_atom37(atom14_pos, residx_atom37_to_atom14, atom37_atom_exists):
    """
    Converts atom14 atomic positions [*, N, 14, 3] to standard atom37 positions [*, N, 37, 3].
    """
    target_shape = tuple(ops.shape(residx_atom37_to_atom14)) + (3,)
    gather_idx = ops.broadcast_to(ops.expand_dims(residx_atom37_to_atom14, axis=-1), target_shape)
    atom37_data = ops.take_along_axis(atom14_pos, gather_idx, axis=-2)
    atom37_data = atom37_data * ops.expand_dims(atom37_atom_exists, axis=-1)
    return atom37_data

