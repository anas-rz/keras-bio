"""Unit tests for ProtT5 layers in keras-bio."""

import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_bio.protein.prot_t5.layers import (
    ProtT5Attention,
    ProtT5CrossAttentionLayer,
    ProtT5DecoderBlock,
    ProtT5DenseActDense,
    ProtT5DenseGatedDense,
    ProtT5EncoderBlock,
    ProtT5FeedForwardLayer,
    ProtT5LayerNorm,
    ProtT5PositionBias,
    ProtT5SelfAttentionLayer,
    ProtT5TiedHead,
    compute_relative_bias,
    relative_position_bucket,
)


def test_layer_norm():
    ln = ProtT5LayerNorm(eps=1e-6)
    x = np.random.randn(2, 5, 32).astype("float32")
    out = ln(x)
    assert out.shape == (2, 5, 32)

    # Verify RMS calculation manually: x / sqrt(mean(x^2) + eps)
    expected_scale = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(ops.convert_to_numpy(out), x * expected_scale, rtol=1e-5, atol=1e-5)

    # Serialization
    config = ln.get_config()
    assert config["eps"] == 1e-6
    reconstructed = ProtT5LayerNorm.from_config(config)
    assert reconstructed.eps == 1e-6


def test_relative_position_bucket():
    rel_pos = ops.convert_to_tensor(np.array([-10, -1, 0, 1, 10, 50, 200]), dtype="int32")
    buckets = relative_position_bucket(rel_pos, bidirectional=True, num_buckets=32, max_distance=128)
    b_np = ops.convert_to_numpy(buckets)
    assert np.all(b_np >= 0)
    assert np.all(b_np < 32)


def test_position_bias():
    rel_emb = layers.Embedding(32, 4)
    rel_emb.build((32,))
    pos_bias = ProtT5PositionBias(
        num_heads=4,
        num_buckets=32,
        max_distance=128,
        bidirectional=True,
        causal=False,
        rel_bias_embedding=rel_emb,
    )
    dummy_hidden = ops.zeros((2, 6, 16), dtype="float32")
    attn_mask = ops.convert_to_tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype="int32")
    bias = pos_bias(dummy_hidden, attention_mask=attn_mask)
    assert bias.shape == (2, 4, 6, 6)
    bias_np = ops.convert_to_numpy(bias)
    # Masked token position in first row should be heavily penalized (< -1e8)
    assert bias_np[0, 0, 0, 5] < -1e8
    assert bias_np[1, 0, 0, 5] > -1e8


def test_attention_self_and_cross():
    attn = ProtT5Attention(embed_dim=32, key_value_dim=8, num_heads=4, dropout=0.0)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 32).astype("float32"))
    pos_bias = ops.convert_to_tensor(np.zeros((1, 4, 5, 5), dtype="float32"))

    # Self-attention
    out_self = attn(x, position_bias=pos_bias)
    assert out_self.shape == (2, 5, 32)

    # Cross-attention
    kv = ops.convert_to_tensor(np.random.randn(2, 7, 32).astype("float32"))
    cross_bias = ops.convert_to_tensor(np.zeros((1, 4, 5, 7), dtype="float32"))
    out_cross = attn(x, position_bias=cross_bias, key_value_states=kv)
    assert out_cross.shape == (2, 5, 32)


def test_dense_act_dense_ungated():
    ff = ProtT5DenseActDense(embed_dim=32, mlp_dim=64, hidden_act="relu", dropout=0.0)
    x = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype("float32"))
    out = ff(x)
    assert out.shape == (2, 4, 32)


def test_dense_gated_dense():
    ff = ProtT5DenseGatedDense(embed_dim=32, mlp_dim=64, hidden_act="gelu_new", dropout=0.0)
    x = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype("float32"))
    out = ff(x)
    assert out.shape == (2, 4, 32)


def test_feed_forward_wrapper():
    # Test both ungated and gated
    ff_ungated = ProtT5FeedForwardLayer(embed_dim=32, mlp_dim=64, is_gated_act=False, dropout=0.0)
    ff_gated = ProtT5FeedForwardLayer(embed_dim=32, mlp_dim=64, is_gated_act=True, hidden_act="gelu_new", dropout=0.0)

    x = ops.convert_to_tensor(np.random.randn(2, 4, 32).astype("float32"))
    assert ff_ungated(x).shape == (2, 4, 32)
    assert ff_gated(x).shape == (2, 4, 32)


def test_encoder_and_decoder_blocks():
    enc_block = ProtT5EncoderBlock(
        embed_dim=32,
        key_value_dim=8,
        num_heads=4,
        mlp_dim=64,
        hidden_act="relu",
        is_gated_act=False,
        dropout=0.0,
    )
    x = ops.convert_to_tensor(np.random.randn(2, 5, 32).astype("float32"))
    pos_bias = ops.convert_to_tensor(np.zeros((1, 4, 5, 5), dtype="float32"))
    enc_out = enc_block(x, position_bias=pos_bias)
    assert enc_out.shape == (2, 5, 32)

    dec_block = ProtT5DecoderBlock(
        embed_dim=32,
        key_value_dim=8,
        num_heads=4,
        mlp_dim=64,
        hidden_act="relu",
        is_gated_act=False,
        dropout=0.0,
    )
    dec_out = dec_block(
        x,
        self_position_bias=pos_bias,
        encoder_hidden_states=enc_out,
        cross_position_bias=pos_bias,
    )
    assert dec_out.shape == (2, 5, 32)


def test_tied_head():
    tied = ProtT5TiedHead()
    hidden = ops.convert_to_tensor(np.random.randn(2, 5, 32).astype("float32"))
    emb_weights = ops.convert_to_tensor(np.random.randn(128, 32).astype("float32"))
    logits = tied([hidden, emb_weights])
    assert logits.shape == (2, 5, 128)

