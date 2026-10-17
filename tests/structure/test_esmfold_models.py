"""Unit and integration tests for ESMFoldModel in keras-bio."""

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


def test_config_serialization(tiny_config):
    d = tiny_config.to_dict()
    assert isinstance(d, dict)
    assert d["esm_hidden_size"] == 64
    assert d["trunk"]["num_blocks"] == 2

    restored = ESMFoldConfig.from_dict(d)
    assert restored.esm_hidden_size == 64
    assert restored.trunk.num_blocks == 2


def test_model_forward_pass(tiny_config):
    model = ESMFoldModel(tiny_config)
    input_ids = ops.convert_to_tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((2, 5), dtype="float32"))

    out = model(input_ids, attention_mask=mask, num_recycles=0)

    assert "positions" in out
    assert "frames" in out
    assert "plddt" in out
    assert "ptm" in out
    assert "distogram_logits" in out
    assert "atom37_positions" in out

    plddt_np = ops.convert_to_numpy(out["plddt"])
    ptm_np = ops.convert_to_numpy(out["ptm"])
    atom37_np = ops.convert_to_numpy(out["atom37_positions"])

    assert plddt_np.shape == (2, 5, 37)
    assert ptm_np.shape == (2,)
    assert atom37_np.shape == (2, 5, 37, 3)
    assert not np.isnan(plddt_np).any()
    assert not np.isnan(ptm_np).any()
    assert not np.isnan(atom37_np).any()


def test_model_infer_and_pdb_generation(tiny_config):
    model = ESMFoldModel(tiny_config)
    seqs = ["MKTVRQE", "ACDEFGHIK"]

    outputs = model.infer(seqs, num_recycles=0)
    assert "pdbs" in outputs
    assert len(outputs["pdbs"]) == 2

    pdb_0 = outputs["pdbs"][0]
    assert isinstance(pdb_0, str)
    lines_0 = pdb_0.strip().split("\n")
    assert lines_0[0].startswith("ATOM")
    assert lines_0[-1] == "END"

    pdb_1 = outputs["pdbs"][1]
    assert isinstance(pdb_1, str)
    lines_1 = pdb_1.strip().split("\n")
    assert lines_1[0].startswith("ATOM")
    assert lines_1[-1] == "END"


def test_protein_to_pdb_util():
    prot = ProteinStructure(
        aatype=np.array([0, 1, 2]),
        atom_positions=np.zeros((3, 37, 3)),
        atom_mask=np.ones((3, 37)),
        residue_index=np.array([1, 2, 3]),
        b_factors=np.array([85.0, 90.0, 75.0]),
        chain_tag="A",
    )
    pdb_txt = protein_to_pdb(prot)
    assert "ALA A   1" in pdb_txt
    assert "ARG A   2" in pdb_txt
    assert "ASN A   3" in pdb_txt
    assert "END" in pdb_txt


def test_model_save_load_roundtrip(tiny_config):
    model = ESMFoldModel(tiny_config)
    input_ids = ops.convert_to_tensor(np.array([[1, 2, 3, 4]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 4), dtype="float32"))

    out_before = ops.convert_to_numpy(model(input_ids, attention_mask=mask, num_recycles=0)["plddt"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "esmfold_tiny.keras")
        model.save(path)

        loaded = keras.models.load_model(path)
        out_after = ops.convert_to_numpy(loaded(input_ids, attention_mask=mask, num_recycles=0)["plddt"])

        np.testing.assert_allclose(out_before, out_after, atol=1e-5)
