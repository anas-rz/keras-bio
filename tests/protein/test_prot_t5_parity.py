"""Exact numerical parity verification against Hugging Face PyTorch T5."""

import numpy as np
import pytest
import torch
from transformers import (
    T5Config as HFT5Config,
    T5EncoderModel as HFT5EncoderModel,
    T5ForConditionalGeneration as HFT5ForConditionalGeneration,
)

from keras_bio.protein import (
    ProtT5Config,
    ProtT5EncoderModel,
    ProtT5ForConditionalGeneration,
    transfer_t5_weights,
)


def test_ungated_encoder_parity():
    """Verify ProtT5EncoderModel matches HF T5EncoderModel with ungated ReLU feed-forward."""
    hf_cfg = HFT5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=32,
        dropout_rate=0.0,
        dense_act_fn="relu",
        is_gated_act=False,
        feed_forward_proj="relu",
    )
    torch_model = HFT5EncoderModel(hf_cfg).eval()
    k_cfg = ProtT5Config.from_hf(hf_cfg.to_dict())
    k_model = ProtT5EncoderModel(k_cfg)
    k_model.build()

    transfer_t5_weights(k_model, torch_model.state_dict())

    np.random.seed(42)
    input_ids = np.random.randint(3, 24, (2, 8)).astype("int32")
    attention_mask = np.ones((2, 8), dtype="int32")
    attention_mask[0, 6:] = 0

    with torch.no_grad():
        pt_out = torch_model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        ).last_hidden_state.numpy()

    k_out = k_model({"input_ids": input_ids, "attention_mask": attention_mask})[
        "last_hidden_state"
    ]
    k_out_np = np.asarray(k_out)

    diff = np.max(np.abs(pt_out - k_out_np))
    assert diff < 1e-4, f"Ungated encoder max diff too large: {diff}"


def test_gated_encoder_parity():
    """Verify ProtT5EncoderModel matches HF T5EncoderModel with gated GELU feed-forward."""
    hf_cfg = HFT5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=32,
        dropout_rate=0.0,
        dense_act_fn="gelu_new",
        is_gated_act=True,
        feed_forward_proj="gated-gelu",
    )
    torch_model = HFT5EncoderModel(hf_cfg).eval()
    k_cfg = ProtT5Config.from_hf(hf_cfg.to_dict())
    k_model = ProtT5EncoderModel(k_cfg)
    k_model.build()

    transfer_t5_weights(k_model, torch_model.state_dict())

    np.random.seed(42)
    input_ids = np.random.randint(3, 24, (2, 8)).astype("int32")
    attention_mask = np.ones((2, 8), dtype="int32")
    attention_mask[0, 6:] = 0

    with torch.no_grad():
        pt_out = torch_model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        ).last_hidden_state.numpy()

    k_out = k_model({"input_ids": input_ids, "attention_mask": attention_mask})[
        "last_hidden_state"
    ]
    k_out_np = np.asarray(k_out)

    diff = np.max(np.abs(pt_out - k_out_np))
    assert diff < 1e-4, f"Gated encoder max diff too large: {diff}"


def test_full_model_parity():
    """Verify ProtT5ForConditionalGeneration matches HF T5ForConditionalGeneration."""
    hf_cfg = HFT5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=32,
        dropout_rate=0.0,
        dense_act_fn="relu",
        is_gated_act=False,
        feed_forward_proj="relu",
        tie_word_embeddings=True,
    )
    torch_model = HFT5ForConditionalGeneration(hf_cfg).eval()
    k_cfg = ProtT5Config.from_hf(hf_cfg.to_dict())
    k_model = ProtT5ForConditionalGeneration(k_cfg)
    k_model.build()

    transfer_t5_weights(k_model, torch_model.state_dict())

    np.random.seed(42)
    input_ids = np.random.randint(3, 24, (2, 8)).astype("int32")
    attention_mask = np.ones((2, 8), dtype="int32")
    attention_mask[0, 6:] = 0
    decoder_input_ids = np.random.randint(3, 24, (2, 5)).astype("int32")

    with torch.no_grad():
        pt_logits = torch_model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            decoder_input_ids=torch.tensor(decoder_input_ids, dtype=torch.long),
        ).logits.numpy()

    k_out = k_model(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }
    )
    k_logits = np.asarray(k_out["logits"])

    diff = np.max(np.abs(pt_logits - k_logits))
    assert diff < 1e-4, f"Full model logits max diff too large: {diff}"

