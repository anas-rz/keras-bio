"""Multi-backend verification for ProtT5 in Keras 3."""

import os
import tempfile
import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.protein import ProtT5Config, ProtT5EncoderModel, ProtT5Model


@pytest.fixture
def tiny_config():
    return ProtT5Config(
        vocab_size=128,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=32,
        dropout_rate=0.0,
    )


def test_backend_execution(tiny_config):
    """Verify forward pass runs properly in current backend."""
    current_backend = keras.backend.backend()
    assert current_backend in ("tensorflow", "torch", "jax")

    model = ProtT5EncoderModel(tiny_config)
    input_ids = ops.convert_to_tensor([[3, 4, 5, 1], [6, 7, 8, 1]], dtype="int32")
    out = model(input_ids)
    assert out["last_hidden_state"].shape == (2, 4, 32)


def test_serialization_roundtrip(tiny_config):
    """Verify model can be saved to .keras and loaded back."""
    model = ProtT5EncoderModel(tiny_config)
    input_ids = np.array([[3, 4, 5, 1], [6, 7, 8, 1]], dtype="int32")
    out_orig = ops.convert_to_numpy(model(input_ids)["last_hidden_state"])

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "prott5_enc.keras")
        model.save(save_path)

        loaded_model = keras.saving.load_model(save_path)
        out_loaded = ops.convert_to_numpy(loaded_model(input_ids)["last_hidden_state"])

        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-5)


def test_embed_pipeline(tiny_config):
    """Verify end-to-end embedding pipeline with sequence strings."""
    model = ProtT5EncoderModel(tiny_config)
    sequences = ["MKLLVL", "VLSACD"]
    per_protein = model.embed(sequences, pool="mean")
    assert per_protein.shape == (2, 32)

    per_residue = model.embed(sequences, pool="none")
    # 6 residues + 1 EOS = 7
    assert per_residue.shape == (2, 7, 32)

