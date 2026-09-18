"""Unit tests for ESM-2 layers in keras-bio."""

import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_bio.protein.esm2.config import ESM2Config
from keras_bio.protein.esm2.layers import (
    ESM2Attention,
    ESM2ClassificationHead,
    ESM2ContactPredictionHead,
    ESM2EncoderLayer,
    ESM2FeedForward,
    ESM2LMHead,
    ESM2RotaryEmbedding,
    ESM2SelfAttention,
    apply_rotary_pos_emb,
    average_product_correct,
    rotate_half,
    symmetrize,
)


def test_rotate_half():
    x = ops.convert_to_tensor(np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype="float32"))
    rotated = rotate_half(x)
    r_np = ops.convert_to_numpy(rotated)
    np.testing.assert_allclose(r_np, np.array([[[-3.0, -4.0, 1.0, 2.0]]]), atol=1e-6)


def test_apply_rotary_pos_emb():
    # q: (batch=1, heads=1, seq=1, dim=4)
    q = ops.convert_to_tensor(np.array([[[[1.0, 0.0, 0.0, 1.0]]]], dtype="float32"))
    k = ops.convert_to_tensor(np.array([[[[0.0, 1.0, 1.0, 0.0]]]], dtype="float32"))
    # cos and sin: (batch=1, seq=1, dim=4)
    cos = ops.convert_to_tensor(np.array([[[1.0, 1.0, 1.0, 1.0]]], dtype="float32"))
    sin = ops.convert_to_tensor(np.array([[[0.0, 0.0, 0.0, 0.0]]], dtype="float32"))

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    np.testing.assert_allclose(ops.convert_to_numpy(q_rot), ops.convert_to_numpy(q), atol=1e-6)
    np.testing.assert_allclose(ops.convert_to_numpy(k_rot), ops.convert_to_numpy(k), atol=1e-6)


def test_symmetrize_and_apc():
    # x: (batch=1, channels=1, seq=3, seq=3)
    mat = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype="float32")
    x = ops.convert_to_tensor(mat)
    sym = symmetrize(x)
    sym_np = ops.convert_to_numpy(sym)
    # Check symmetry: mat + mat.T
    np.testing.assert_allclose(sym_np, sym_np.transpose(0, 1, 3, 2), atol=1e-6)

    apc = average_product_correct(sym)
    apc_np = ops.convert_to_numpy(apc)
    assert apc_np.shape == (1, 1, 3, 3)


def test_rotary_embedding_layer():
    rot = ESM2RotaryEmbedding(dim=16, base=10000.0)
    x = ops.convert_to_tensor(np.zeros((2, 8, 64), dtype="float32"))
    cos, sin = rot(x)
    assert cos.shape == (2, 8, 16)
    assert sin.shape == (2, 8, 16)

    # Serialization
    config = rot.get_config()
    reconstructed = ESM2RotaryEmbedding.from_config(config)
    assert reconstructed.dim == 16
    assert reconstructed.base == 10000.0


def test_self_attention_layer():
    cfg = ESM2Config(hidden_size=64, num_attention_heads=4)
    attn = ESM2SelfAttention(cfg)
    x = ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32"))
    rot = ESM2RotaryEmbedding(dim=cfg.head_dim, base=cfg.rope_theta)
    cos, sin = rot(x)

    out = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 6, 64)

    # Output attentions
    out, weights = attn(x, position_embeddings=(cos, sin), output_attentions=True)
    assert out.shape == (2, 6, 64)
    assert weights.shape == (2, 4, 6, 6)

    # Serialization
    config = attn.get_config()
    reconstructed = ESM2SelfAttention.from_config(config)
    assert reconstructed.hidden_size == 64
    assert reconstructed.num_attention_heads == 4


def test_attention_layer():
    cfg = ESM2Config(hidden_size=64, num_attention_heads=4)
    attn_layer = ESM2Attention(cfg)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))
    rot = ESM2RotaryEmbedding(dim=cfg.head_dim, base=cfg.rope_theta)
    cos, sin = rot(x)

    out = attn_layer(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 5, 64)

    # Serialization
    config = attn_layer.get_config()
    reconstructed = ESM2Attention.from_config(config)
    assert reconstructed.config.hidden_size == 64


def test_feed_forward_layer():
    cfg = ESM2Config(hidden_size=64, intermediate_size=128)
    ff = ESM2FeedForward(cfg)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))
    out = ff(x)
    assert out.shape == (2, 5, 64)

    # Serialization
    config = ff.get_config()
    reconstructed = ESM2FeedForward.from_config(config)
    assert reconstructed.config.intermediate_size == 128


def test_encoder_layer():
    cfg = ESM2Config(hidden_size=64, num_attention_heads=4, intermediate_size=128)
    enc = ESM2EncoderLayer(cfg)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))
    rot = ESM2RotaryEmbedding(dim=cfg.head_dim, base=cfg.rope_theta)
    cos, sin = rot(x)

    out = enc(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 5, 64)

    out, weights = enc(x, position_embeddings=(cos, sin), output_attentions=True)
    assert out.shape == (2, 5, 64)
    assert weights.shape == (2, 4, 5, 5)


def test_contact_prediction_head():
    # in_features = 2 layers * 4 heads = 8
    head = ESM2ContactPredictionHead(in_features=8, eos_idx=2)
    tokens = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 2, 1], [0, 7, 8, 9, 10, 2]], dtype="int32"))
    attentions = ops.convert_to_tensor(np.random.rand(2, 2, 4, 6, 6).astype("float32"))

    contacts = head(tokens, attentions)
    # Contact map shape is (batch, seq_len - 2, seq_len - 2) = (2, 4, 4)
    assert contacts.shape == (2, 4, 4)
    c_np = ops.convert_to_numpy(contacts)
    # Check probabilities are in [0, 1]
    assert np.all(c_np >= 0.0)
    assert np.all(c_np <= 1.0)


def test_lm_head():
    cfg = ESM2Config(vocab_size=33, hidden_size=64)
    emb_weights = ops.convert_to_tensor(np.random.randn(33, 64).astype("float32"))
    head = ESM2LMHead(cfg, tied_embedding_weights=emb_weights)

    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))
    logits = head(x)
    assert logits.shape == (2, 5, 33)


def test_classification_head():
    cfg = ESM2Config(hidden_size=64, num_labels=3)
    head = ESM2ClassificationHead(cfg)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))
    logits = head(x)
    assert logits.shape == (2, 3)

