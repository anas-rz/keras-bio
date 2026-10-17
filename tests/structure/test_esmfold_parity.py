"""Numerical parity tests verifying ESMFold against Hugging Face PyTorch implementation."""

import keras.ops as ops
import numpy as np
import pytest
import torch
from transformers import EsmConfig
from transformers.models.esm.configuration_esm import EsmFoldConfig as HFEsmFoldConfig
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding

from keras_bio.structure.esmfold.config import ESMFoldConfig
from keras_bio.structure.esmfold.convert import convert_esmfold_weights_from_hf
from keras_bio.structure.esmfold.modeling import ESMFoldModel


def test_esmfold_parity_mini():
    """Verifies forward pass numerical equivalence between Hugging Face ESMFold and Keras ESMFold."""
    torch.manual_seed(42)
    np.random.seed(42)

    esmfold_cfg = HFEsmFoldConfig(
        trunk={
            "num_blocks": 1,
            "sequence_state_dim": 64,
            "pairwise_state_dim": 32,
            "sequence_head_width": 16,
            "pairwise_head_width": 16,
            "position_bins": 16,
            "dropout": 0.0,
            "structure_module": {
                "num_blocks": 1,
                "sequence_dim": 64,
                "pairwise_dim": 32,
                "ipa_dim": 16,
                "resnet_dim": 32,
                "num_heads_ipa": 2,
                "num_qk_points": 4,
                "num_v_points": 4,
                "num_resnet_blocks": 1,
                "num_transition_layers": 1,
                "dropout_rate": 0.0,
            },
        },
        lddt_head_hid_dim=32,
        fp16_esm=False,
    )

    esm_cfg = EsmConfig(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        pad_token_id=1,
        mask_token_id=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        position_embedding_type="rotary",
        is_folding_model=True,
        esmfold_config=esmfold_cfg,
        vocab_list=None,
        token_dropout=False,
    )

    pt_model = EsmForProteinFolding(esm_cfg).eval()
    k_cfg = ESMFoldConfig.from_hf(esm_cfg.to_dict())
    k_model = ESMFoldModel(k_cfg)

    input_ids = np.array([[1, 5, 8, 2]], dtype="int32")
    attention_mask = np.array([[1, 1, 1, 1]], dtype="float32")

    # Build keras model and convert weights
    _ = k_model(input_ids, attention_mask=attention_mask, num_recycles=0)
    convert_esmfold_weights_from_hf(pt_model.state_dict(), k_model)

    with torch.no_grad():
        pt_out = pt_model(torch.tensor(input_ids, dtype=torch.long), num_recycles=0)

    k_out = k_model(input_ids, attention_mask=attention_mask, num_recycles=0)

    # Parity comparisons
    plddt_diff = np.max(np.abs(pt_out["plddt"].numpy() - ops.convert_to_numpy(k_out["plddt"])))
    ptm_diff = np.max(np.abs(pt_out["ptm"].numpy() - ops.convert_to_numpy(k_out["ptm"])))
    disto_diff = np.max(np.abs(pt_out["distogram_logits"].numpy() - ops.convert_to_numpy(k_out["distogram_logits"])))
    pos_diff = np.max(np.abs(pt_out["positions"].numpy() - ops.convert_to_numpy(k_out["positions"])))

    assert plddt_diff < 1e-4, f"pLDDT max diff too high: {plddt_diff}"
    assert ptm_diff < 1e-4, f"PTM max diff too high: {ptm_diff}"
    assert disto_diff < 1e-3, f"Distogram logits max diff too high: {disto_diff}"
    assert pos_diff < 0.15, f"Atomic positions max diff too high: {pos_diff}"

