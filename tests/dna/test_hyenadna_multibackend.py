"""Multi-backend execution and serialization tests for HyenaDNA."""

import os
import tempfile

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.dna.hyena import (
    HyenaDNAConfig,
    HyenaDNAForCausalLM,
    HyenaDNAForSequenceClassification,
    HyenaDNAModel,
    HyenaDNATokenizer,
)


@pytest.fixture
def hyena_config():
    return HyenaDNAConfig(
        d_model=64,
        n_layer=2,
        d_inner=128,
        vocab_size=12,
        pad_vocab_size_multiple=8,
        max_seq_len=64,
        filter_order=32,
        short_filter_order=3,
        emb_dim=5,
        hyena_order=2,
        activation_freq=10.0,
        num_inner_mlps=2,
        layer_norm_epsilon=1e-5,
    )


def test_multibackend_forward(hyena_config):
    """Test HyenaDNAModel forward pass on current backend."""
    model = HyenaDNAModel(hyena_config)
    input_ids = ops.convert_to_tensor(np.array([[7, 8, 9, 10, 1]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    out = model(input_ids, attention_mask=mask)
    assert "last_hidden_state" in out
    hidden = ops.convert_to_numpy(out["last_hidden_state"])
    assert hidden.shape == (1, 5, 64)
    assert not np.isnan(hidden).any()


def test_multibackend_causal_lm(hyena_config):
    """Test HyenaDNAForCausalLM forward pass on current backend."""
    model = HyenaDNAForCausalLM(hyena_config)
    input_ids = ops.convert_to_tensor(np.array([[7, 8, 9, 10, 1]], dtype="int32"))

    out = model(input_ids)
    assert "logits" in out
    logits = ops.convert_to_numpy(out["logits"])
    assert logits.shape == (1, 5, 16)
    assert not np.isnan(logits).any()


def test_multibackend_sequence_classification(hyena_config):
    """Test HyenaDNAForSequenceClassification on current backend."""
    model = HyenaDNAForSequenceClassification(hyena_config, num_labels=3)
    input_ids = ops.convert_to_tensor(np.array([[7, 8, 9, 10, 1]], dtype="int32"))

    out = model(input_ids)
    assert "logits" in out
    logits = ops.convert_to_numpy(out["logits"])
    assert logits.shape == (1, 3)
    assert not np.isnan(logits).any()


def test_multibackend_embed_api(hyena_config):
    """Test embed() method with DNA sequence strings."""
    model = HyenaDNAModel(hyena_config)
    tok = HyenaDNATokenizer()
    seqs = ["ATCGATCG", "GCTAGCTA"]

    mean_pooled = model.embed(seqs, tokenizer=tok, pool="mean")
    assert isinstance(mean_pooled, np.ndarray)
    assert mean_pooled.shape == (2, 64)
    assert not np.isnan(mean_pooled).any()

    last_pooled = model.embed(seqs, tokenizer=tok, pool="last")
    assert isinstance(last_pooled, np.ndarray)
    assert last_pooled.shape == (2, 64)
    assert not np.isnan(last_pooled).any()


def test_multibackend_save_load_roundtrip(hyena_config):
    """Verify HyenaDNA model saves to .keras and restores exact weights and outputs."""
    model = HyenaDNAModel(hyena_config)
    input_ids = ops.convert_to_tensor(np.array([[7, 8, 9, 10, 1]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    out_before = ops.convert_to_numpy(model(input_ids, attention_mask=mask)["last_hidden_state"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_hyenadna.keras")
        model.save(path)

        loaded = keras.models.load_model(path)
        out_after = ops.convert_to_numpy(loaded(input_ids, attention_mask=mask)["last_hidden_state"])

        np.testing.assert_allclose(out_before, out_after, atol=1e-6)

