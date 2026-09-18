"""Multi-backend execution and serialization tests for ESM-2."""

import os
import tempfile

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.protein.esm2 import (
    ESM2Config,
    ESM2ForMaskedLM,
    ESM2Model,
    ESMTokenizer,
)


@pytest.fixture
def esm2_config():
    return ESM2Config(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def test_multibackend_forward(esm2_config):
    """Test ESM2 forward pass runs without error on current backend."""
    model = ESM2Model(esm2_config)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 2]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    out = model(input_ids, attention_mask=mask)
    assert "last_hidden_state" in out
    hidden = ops.convert_to_numpy(out["last_hidden_state"])
    assert hidden.shape == (1, 5, 64)
    assert not np.isnan(hidden).any()


def test_multibackend_contact_prediction(esm2_config):
    """Test contact map prediction runs and produces valid probabilities."""
    model = ESM2Model(esm2_config)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 7, 2]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 6), dtype="int32"))

    contacts = model.predict_contacts(input_ids, attention_mask=mask)
    c_np = ops.convert_to_numpy(contacts)
    assert c_np.shape == (1, 4, 4)
    assert not np.isnan(c_np).any()
    assert np.all(c_np >= 0.0)
    assert np.all(c_np <= 1.0)


def test_multibackend_embed_api(esm2_config):
    """Test embed() method with sequence strings."""
    model = ESM2Model(esm2_config)
    tokenizer = ESMTokenizer()
    seqs = ["MKTV", "SEQWENCE"]

    pooled = model.embed(seqs, tokenizer=tokenizer, pool="mean")
    assert isinstance(pooled, np.ndarray)
    assert pooled.shape == (2, 64)
    assert not np.isnan(pooled).any()


def test_multibackend_save_load_roundtrip(esm2_config):
    """Verify ESM2 model saves to .keras and restores exact weights and outputs."""
    model = ESM2Model(esm2_config)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 2]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    out_before = ops.convert_to_numpy(model(input_ids, attention_mask=mask)["last_hidden_state"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_esm2.keras")
        model.save(path)

        loaded = keras.models.load_model(path)
        out_after = ops.convert_to_numpy(loaded(input_ids, attention_mask=mask)["last_hidden_state"])

        np.testing.assert_allclose(out_before, out_after, atol=1e-6)

