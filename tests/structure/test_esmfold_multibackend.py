"""Multi-backend execution and serialization tests for ESMFold."""

import os
import tempfile

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.structure.esmfold import (
    ESMFoldConfig,
    ESMFoldModel,
    ProteinStructure,
    protein_to_pdb,
)


@pytest.fixture
def tiny_config():
    return ESMFoldConfig.from_preset("esmfold_tiny")


def test_multibackend_forward(tiny_config):
    """Verifies ESMFoldModel forward pass across backends."""
    model = ESMFoldModel(tiny_config)
    input_ids = ops.convert_to_tensor(np.array([[1, 2, 3, 4]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 4), dtype="float32"))

    out = model(input_ids, attention_mask=mask, num_recycles=0)

    assert "positions" in out
    assert "plddt" in out
    assert "ptm" in out
    assert "distogram_logits"
    assert "atom37_positions" in out

    positions_np = ops.convert_to_numpy(out["positions"])
    plddt_np = ops.convert_to_numpy(out["plddt"])
    ptm_np = ops.convert_to_numpy(out["ptm"])
    atom37_np = ops.convert_to_numpy(out["atom37_positions"])

    assert positions_np.shape == (1, tiny_config.trunk.structure_module.num_blocks, 4, 14, 3)
    assert plddt_np.shape == (1, 4, 37)
    assert ptm_np.shape == (1,)
    assert atom37_np.shape == (1, 4, 37, 3)

    assert not np.isnan(positions_np).any()
    assert not np.isnan(plddt_np).any()
    assert not np.isnan(atom37_np).any()


def test_multibackend_infer_and_pdb(tiny_config):
    """Verifies high-level string sequence inference and PDB output across backends."""
    model = ESMFoldModel(tiny_config)
    res = model.infer("MLKN", num_recycles=0)

    assert "pdbs" in res
    assert len(res["pdbs"]) == 1
    pdb_str = res["pdbs"][0]
    assert "ATOM" in pdb_str
    assert "CA" in pdb_str
    assert "END" in pdb_str


def test_multibackend_save_load_roundtrip(tiny_config):
    """Verifies Keras 3 .keras format saving and loading roundtrip."""
    model = ESMFoldModel(tiny_config)
    input_ids = ops.convert_to_tensor(np.array([[1, 2, 3, 4]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 4), dtype="float32"))

    out_before = ops.convert_to_numpy(model(input_ids, attention_mask=mask, num_recycles=0)["plddt"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "esmfold_tiny_mb.keras")
        model.save(path)

        loaded = keras.models.load_model(path)
        out_after = ops.convert_to_numpy(loaded(input_ids, attention_mask=mask, num_recycles=0)["plddt"])

        np.testing.assert_allclose(out_before, out_after, atol=1e-5)

