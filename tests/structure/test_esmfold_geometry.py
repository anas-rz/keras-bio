"""Unit and parity tests for 3D Lie Group geometry primitives in ESMFold."""

import keras
from keras import ops
import numpy as np
import pytest
import torch
from transformers.models.esm.openfold_utils.rigid_utils import (
    Rigid as TorchRigid,
    Rotation as TorchRotation,
)

from keras_bio.structure.esmfold.geometry import (
    Rigid,
    Rotation,
    invert_rot_mat,
    quat_to_rot,
    rot_matmul,
    rot_vec_mul,
)


def test_rotation_identity():
    rot = Rotation.identity(shape=(2, 3))
    rot_mat = ops.convert_to_numpy(rot.get_rot_mats())
    assert rot_mat.shape == (2, 3, 3, 3)
    for b in range(2):
        for i in range(3):
            np.testing.assert_allclose(rot_mat[b, i], np.eye(3), atol=1e-6)


def test_rotation_composition_and_apply():
    # Random quaternion
    q_np = np.random.randn(2, 4, 4).astype(np.float32)
    q_np = q_np / np.linalg.norm(q_np, axis=-1, keepdims=True)
    rot = Rotation(quats=ops.convert_to_tensor(q_np))

    pts_np = np.random.randn(2, 4, 3).astype(np.float32)
    pts = ops.convert_to_tensor(pts_np)

    # Apply then invert_apply recovers pts
    rotated = rot.apply(pts)
    recovered = rot.invert_apply(rotated)
    np.testing.assert_allclose(ops.convert_to_numpy(recovered), pts_np, atol=1e-5)


def test_rigid_identity_and_compose():
    rig = Rigid.identity(shape=(2, 4))
    r_np = ops.convert_to_numpy(rig.get_rots().get_rot_mats())
    t_np = ops.convert_to_numpy(rig.get_trans())
    assert r_np.shape == (2, 4, 3, 3)
    assert t_np.shape == (2, 4, 3)
    np.testing.assert_allclose(t_np, 0.0, atol=1e-6)

    # Random rigid transform
    q_np = np.random.randn(2, 4, 4).astype(np.float32)
    q_np = q_np / np.linalg.norm(q_np, axis=-1, keepdims=True)
    t_np = np.random.randn(2, 4, 3).astype(np.float32)
    pts_np = np.random.randn(2, 4, 3).astype(np.float32)

    rig = Rigid(Rotation(quats=ops.convert_to_tensor(q_np)), ops.convert_to_tensor(t_np))
    pts = ops.convert_to_tensor(pts_np)

    transformed = rig.apply(pts)
    recovered = rig.invert_apply(transformed)
    np.testing.assert_allclose(ops.convert_to_numpy(recovered), pts_np, atol=1e-5)


def test_rigid_compose_q_update():
    rig = Rigid.identity(shape=(2, 3))
    update_vec = ops.convert_to_tensor(np.random.randn(2, 3, 6).astype(np.float32) * 0.1)
    updated_rig = rig.compose_q_update_vec(update_vec)
    assert updated_rig.shape == (2, 3)
    assert ops.convert_to_numpy(updated_rig.get_trans()).shape == (2, 3, 3)


def test_rigid_4x4_conversion():
    m44 = ops.convert_to_tensor(np.random.randn(3, 4, 4, 4).astype(np.float32))
    rig = Rigid.from_tensor_4x4(m44)
    m44_out = rig.to_tensor_4x4()
    assert ops.convert_to_numpy(m44_out).shape == (3, 4, 4, 4)


def test_geometry_parity_with_torch():
    """Verify numerical parity of Rotation and Rigid operations against OpenFold PyTorch."""
    q_np = np.random.randn(2, 4, 4).astype(np.float32)
    q_np = q_np / np.linalg.norm(q_np, axis=-1, keepdims=True)
    trans_np = np.random.randn(2, 4, 3).astype(np.float32)
    pts_np = np.random.randn(2, 4, 3).astype(np.float32)
    update_np = np.random.randn(2, 4, 6).astype(np.float32) * 0.05

    # Torch reference
    t_rot = TorchRotation(quats=torch.from_numpy(q_np))
    t_rig = TorchRigid(t_rot, torch.from_numpy(trans_np))
    t_applied = t_rig.apply(torch.from_numpy(pts_np)).numpy()
    t_inv_applied = t_rig.invert_apply(torch.from_numpy(pts_np)).numpy()
    t_updated = t_rig.compose_q_update_vec(torch.from_numpy(update_np))
    t_updated_trans = t_updated.get_trans().numpy()
    t_updated_rot = t_updated.get_rots().get_rot_mats().numpy()

    # Keras implementation
    k_rot = Rotation(quats=ops.convert_to_tensor(q_np))
    k_rig = Rigid(k_rot, ops.convert_to_tensor(trans_np))
    k_applied = ops.convert_to_numpy(k_rig.apply(ops.convert_to_tensor(pts_np)))
    k_inv_applied = ops.convert_to_numpy(k_rig.invert_apply(ops.convert_to_tensor(pts_np)))
    k_updated = k_rig.compose_q_update_vec(ops.convert_to_tensor(update_np))
    k_updated_trans = ops.convert_to_numpy(k_updated.get_trans())
    k_updated_rot = ops.convert_to_numpy(k_updated.get_rots().get_rot_mats())

    np.testing.assert_allclose(k_applied, t_applied, atol=1e-6)
    np.testing.assert_allclose(k_inv_applied, t_inv_applied, atol=1e-6)
    np.testing.assert_allclose(k_updated_trans, t_updated_trans, atol=1e-6)
    np.testing.assert_allclose(k_updated_rot, t_updated_rot, atol=1e-6)

