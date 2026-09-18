"""Unit tests for HyenaDNA layers in keras-bio."""

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.dna.hyena.config import HyenaDNAConfig
from keras_bio.dna.hyena.layers import (
    HyenaBlock,
    HyenaExponentialModulation,
    HyenaFilter,
    HyenaMlp,
    HyenaOperator,
    HyenaPositionalEmbedding,
    HyenaSin,
    fftconv,
)


@pytest.fixture
def hyena_config():
    return HyenaDNAConfig(
        d_model=64,
        n_layer=2,
        d_inner=128,
        vocab_size=12,
        max_seq_len=64,
        filter_order=32,
        short_filter_order=3,
        emb_dim=5,
        hyena_order=2,
        activation_freq=10.0,
        num_inner_mlps=2,
        layer_norm_epsilon=1e-5,
    )


def test_fftconv():
    u = ops.convert_to_tensor(np.random.randn(2, 8, 16).astype("float32"))
    k = ops.convert_to_tensor(np.random.randn(8, 16).astype("float32"))
    D = ops.convert_to_tensor(np.random.randn(8).astype("float32"))

    out = fftconv(u, k, D)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == (2, 8, 16)
    assert not np.isnan(out_np).any()


def test_hyena_sin(hyena_config):
    sin_layer = HyenaSin(hyena_config)
    x = ops.convert_to_tensor(np.random.randn(2, 10, 32).astype("float32"))
    out = sin_layer(x)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == (2, 10, 32)
    assert np.all(out_np >= -1.0)
    assert np.all(out_np <= 1.0)

    # Serialization
    config = sin_layer.get_config()
    reconstructed = HyenaSin.from_config(config)
    assert reconstructed.filter_order == 32


def test_hyena_pos_emb(hyena_config):
    pos_emb = HyenaPositionalEmbedding(hyena_config)
    pos_emb.build()
    z, t = pos_emb(16)
    z_np = ops.convert_to_numpy(z)
    t_np = ops.convert_to_numpy(t)
    assert z_np.shape == (1, 16, 5)
    assert t_np.shape == (1, 16, 1)

    # Serialization
    config = pos_emb.get_config()
    reconstructed = HyenaPositionalEmbedding.from_config(config)
    assert reconstructed.emb_dim == 5
    assert reconstructed.seq_len == 64


def test_hyena_exp_modulation(hyena_config):
    mod = HyenaExponentialModulation(hyena_config)
    mod.build()
    t = ops.convert_to_tensor(np.linspace(0, 1, 16).astype("float32")[None, :, None])
    x = ops.convert_to_tensor(np.ones((1, 16, 64), dtype="float32"))

    out = mod(t, x)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == (1, 16, 64)
    assert not np.isnan(out_np).any()

    # Serialization
    config = mod.get_config()
    reconstructed = HyenaExponentialModulation.from_config(config)
    assert reconstructed.d_model == 64


def test_hyena_filter(hyena_config):
    flt = HyenaFilter(hyena_config)
    flt.build()
    h = flt.filter(16)
    h_np = ops.convert_to_numpy(h)
    assert h_np.shape == (1, 16, 64)

    # Test call with fftconv
    x = ops.convert_to_tensor(np.random.randn(2, 64, 16).astype("float32"))
    out = flt(x, 16)
    assert out.shape == (2, 64, 16)

    # Serialization
    config = flt.get_config()
    reconstructed = HyenaFilter.from_config(config)
    assert reconstructed.d_model == 64


def test_hyena_operator(hyena_config):
    op = HyenaOperator(hyena_config)
    u = ops.convert_to_tensor(np.random.randn(2, 16, 64).astype("float32"))

    out = op(u)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == (2, 16, 64)
    assert not np.isnan(out_np).any()

    # Serialization
    config = op.get_config()
    reconstructed = HyenaOperator.from_config(config)
    assert reconstructed.d_model == 64


def test_hyena_mlp(hyena_config):
    mlp = HyenaMlp(hyena_config)
    x = ops.convert_to_tensor(np.random.randn(2, 16, 64).astype("float32"))

    out = mlp(x)
    assert out.shape == (2, 16, 64)

    # Serialization
    config = mlp.get_config()
    reconstructed = HyenaMlp.from_config(config)
    assert reconstructed.config.d_inner == 128


def test_hyena_block(hyena_config):
    block = HyenaBlock(hyena_config)
    x = ops.convert_to_tensor(np.random.randn(2, 16, 64).astype("float32"))

    out = block(x)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == (2, 16, 64)
    assert not np.isnan(out_np).any()

    # Serialization
    config = block.get_config()
    reconstructed = HyenaBlock.from_config(config)
    assert reconstructed.config.d_model == 64

