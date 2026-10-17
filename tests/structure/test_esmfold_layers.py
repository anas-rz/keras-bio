"""Unit tests for individual ESMFold layers and modules in keras-bio."""

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.structure.esmfold.config import (
    FoldingTrunkConfig,
    StructureModuleConfig,
)
from keras_bio.structure.esmfold.geometry import Rigid
from keras_bio.structure.esmfold.layers import (
    EsmFoldAngleResnet,
    EsmFoldAttention,
    EsmFoldBackboneUpdate,
    EsmFoldInvariantPointAttention,
    EsmFoldPairToSequence,
    EsmFoldRelativePosition,
    EsmFoldResidueMLP,
    EsmFoldSelfAttention,
    EsmFoldSequenceToPair,
    EsmFoldStructureModule,
    EsmFoldStructureModuleTransition,
    EsmFoldTriangleAttention,
    EsmFoldTriangleMultiplicativeUpdate,
    EsmFoldTriangularSelfAttentionBlock,
    LDDTHead,
)


def test_relative_position_layer():
    layer = EsmFoldRelativePosition(position_bins=8, pairwise_state_dim=16)
    pos = ops.convert_to_tensor(np.array([[0, 1, 2, 3]], dtype="int32"))
    emb = layer(pos)
    assert ops.convert_to_numpy(emb).shape == (1, 4, 4, 16)


def test_self_attention_layer():
    layer = EsmFoldSelfAttention(embed_dim=32, num_heads=4, head_width=8, gated=True)
    x = ops.convert_to_tensor(np.random.randn(2, 6, 32).astype(np.float32))
    mask = ops.convert_to_tensor(np.ones((2, 6), dtype=np.float32))
    bias = ops.convert_to_tensor(np.zeros((2, 6, 6, 4), dtype=np.float32))

    out, attn_probs = layer(x, mask=mask, bias=bias)
    assert ops.convert_to_numpy(out).shape == (2, 6, 32)
    assert ops.convert_to_numpy(attn_probs).shape == (2, 4, 6, 6)


def test_sequence_to_pair_layer():
    layer = EsmFoldSequenceToPair(sequence_state_dim=32, inner_dim=8, pairwise_state_dim=16)
    s = ops.convert_to_tensor(np.random.randn(2, 5, 32).astype(np.float32))
    z = layer(s)
    assert ops.convert_to_numpy(z).shape == (2, 5, 5, 16)


def test_pair_to_sequence_layer():
    layer = EsmFoldPairToSequence(pairwise_state_dim=16, num_heads=4)
    z = ops.convert_to_tensor(np.random.randn(2, 5, 5, 16).astype(np.float32))
    bias = layer(z)
    assert ops.convert_to_numpy(bias).shape == (2, 5, 5, 4)


def test_residue_mlp_layer():
    layer = EsmFoldResidueMLP(embed_dim=32, inner_dim=64)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 32).astype(np.float32))
    out = layer(x)
    assert ops.convert_to_numpy(out).shape == (2, 5, 32)


def test_triangle_multiplicative_update_layer():
    layer_out = EsmFoldTriangleMultiplicativeUpdate(pairwise_state_dim=16, outgoing=True)
    layer_in = EsmFoldTriangleMultiplicativeUpdate(pairwise_state_dim=16, outgoing=False)

    z = ops.convert_to_tensor(np.random.randn(2, 4, 4, 16).astype(np.float32))
    out1 = layer_out(z)
    out2 = layer_in(z)

    assert ops.convert_to_numpy(out1).shape == (2, 4, 4, 16)
    assert ops.convert_to_numpy(out2).shape == (2, 4, 4, 16)


def test_triangle_attention_layer():
    layer_start = EsmFoldTriangleAttention(c_in=16, c_hidden=8, no_heads=2, starting=True)
    layer_end = EsmFoldTriangleAttention(c_in=16, c_hidden=8, no_heads=2, starting=False)

    z = ops.convert_to_tensor(np.random.randn(2, 4, 4, 16).astype(np.float32))
    out_start = layer_start(z)
    out_end = layer_end(z)

    assert ops.convert_to_numpy(out_start).shape == (2, 4, 4, 16)
    assert ops.convert_to_numpy(out_end).shape == (2, 4, 4, 16)


def test_triangular_self_attention_block():
    cfg = FoldingTrunkConfig(
        num_blocks=1,
        sequence_state_dim=32,
        pairwise_state_dim=16,
        sequence_head_width=8,
        pairwise_head_width=8,
    )
    block = EsmFoldTriangularSelfAttentionBlock(cfg)

    s = ops.convert_to_tensor(np.random.randn(2, 5, 32).astype(np.float32))
    z = ops.convert_to_tensor(np.random.randn(2, 5, 5, 16).astype(np.float32))
    mask = ops.convert_to_tensor(np.ones((2, 5), dtype=np.float32))

    s_out, z_out = block(s, z, mask=mask)
    assert ops.convert_to_numpy(s_out).shape == (2, 5, 32)
    assert ops.convert_to_numpy(z_out).shape == (2, 5, 5, 16)


def test_invariant_point_attention():
    ipa = EsmFoldInvariantPointAttention(
        sequence_dim=32,
        pairwise_dim=16,
        ipa_dim=8,
        num_heads=2,
        num_qk_points=2,
        num_v_points=2,
    )
    s = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype(np.float32))
    z = ops.convert_to_tensor(np.random.randn(2, 4, 4, 16).astype(np.float32))
    rig = Rigid.identity(shape=(2, 4))
    mask = ops.convert_to_tensor(np.ones((2, 4), dtype=np.float32))

    out = ipa(s, z, rig, mask=mask)
    assert ops.convert_to_numpy(out).shape == (2, 4, 32)


def test_backbone_update_and_angle_resnet():
    bb = EsmFoldBackboneUpdate(sequence_dim=32)
    s = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype(np.float32))
    update_6d = bb(s)
    assert ops.convert_to_numpy(update_6d).shape == (2, 4, 6)

    angle_net = EsmFoldAngleResnet(sequence_dim=32, resnet_dim=16, num_resnet_blocks=1, num_angles=7)
    s_initial = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype(np.float32))
    unnorm, norm = angle_net(s, s_initial)
    assert ops.convert_to_numpy(unnorm).shape == (2, 4, 7, 2)
    assert ops.convert_to_numpy(norm).shape == (2, 4, 7, 2)


def test_structure_module():
    cfg = StructureModuleConfig(
        sequence_dim=32,
        pairwise_dim=16,
        ipa_dim=8,
        num_heads_ipa=2,
        num_qk_points=2,
        num_v_points=2,
        num_blocks=1,
        num_resnet_blocks=1,
        resnet_dim=16,
    )
    sm = EsmFoldStructureModule(cfg)

    s = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype(np.float32))
    z = ops.convert_to_tensor(np.random.randn(2, 4, 4, 16).astype(np.float32))
    aatype = ops.convert_to_tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((2, 4), dtype=np.float32))

    out = sm({"single": s, "pair": z}, aatype, mask=mask)
    assert "positions" in out
    assert "frames" in out
    assert "single" in out
    assert ops.convert_to_numpy(out["positions"]).shape == (2, 1, 4, 14, 3)
    assert ops.convert_to_numpy(out["frames"]).shape == (2, 1, 4, 7)


def test_lddt_head():
    head = LDDTHead(sequence_dim=32, hidden_dim=16, lddt_bins=50)
    s = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype(np.float32))
    logits = head(s)
    assert ops.convert_to_numpy(logits).shape == (2, 4, 37, 50)
