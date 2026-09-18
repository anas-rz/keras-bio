"""Unit tests for ESM-2 models and tokenizer in keras-bio."""

import os
import tempfile

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.protein.esm2.config import ESM2Config
from keras_bio.protein.esm2.modeling import (
    ESM2ForMaskedLM,
    ESM2ForSequenceClassification,
    ESM2ForTokenClassification,
    ESM2Model,
)
from keras_bio.protein.esm2.tokenizer import ESMTokenizer


@pytest.fixture
def small_config():
    return ESM2Config(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=3,
    )


def test_esm2_model_forward_pass(small_config):
    model = ESM2Model(small_config)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 2], [0, 7, 8, 9, 2]], dtype="int32"))
    attention_mask = ops.convert_to_tensor(np.ones((2, 5), dtype="int32"))

    outputs = model(input_ids, attention_mask=attention_mask)
    assert "last_hidden_state" in outputs
    assert outputs["last_hidden_state"].shape == (2, 5, 64)

    # With output hidden states and attentions
    outputs_full = model(
        input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
    )
    assert len(outputs_full["hidden_states"]) == 3  # embeddings + 2 layers
    assert len(outputs_full["attentions"]) == 2     # 2 layers
    assert outputs_full["attentions"][0].shape == (2, 4, 5, 5)


def test_esm2_predict_contacts(small_config):
    model = ESM2Model(small_config)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 7, 2], [0, 8, 9, 10, 11, 2]], dtype="int32"))
    attention_mask = ops.convert_to_tensor(np.ones((2, 6), dtype="int32"))

    contacts = model.predict_contacts(input_ids, attention_mask=attention_mask)
    # Contact map shape is (batch, seq_len - 2, seq_len - 2) = (2, 4, 4)
    assert contacts.shape == (2, 4, 4)
    c_np = ops.convert_to_numpy(contacts)
    assert np.all(c_np >= 0.0)
    assert np.all(c_np <= 1.0)


def test_esm2_embed(small_config):
    model = ESM2Model(small_config)
    tokenizer = ESMTokenizer()
    seqs = ["MKTV", "SEQWENCE"]

    # Mean pooling
    pooled = model.embed(seqs, tokenizer=tokenizer, pool="mean", return_numpy=True)
    assert isinstance(pooled, np.ndarray)
    assert pooled.shape == (2, 64)

    # CLS pooling
    cls_emb = model.embed(seqs, tokenizer=tokenizer, pool="cls", return_numpy=True)
    assert cls_emb.shape == (2, 64)

    # Per-residue embeddings
    per_res = model.embed(seqs, tokenizer=tokenizer, pool=None, return_numpy=True)
    assert per_res.shape[0] == 2
    assert per_res.shape[2] == 64


def test_esm2_for_masked_lm(small_config):
    model = ESM2ForMaskedLM(small_config)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 32, 6, 2]], dtype="int32"))
    attention_mask = ops.convert_to_tensor(np.array([[1, 1, 1, 1, 1]], dtype="int32"))

    out = model({"input_ids": input_ids, "attention_mask": attention_mask})
    assert "logits" in out
    assert out["logits"].shape == (1, 5, 33)


def test_esm2_for_sequence_classification(small_config):
    model = ESM2ForSequenceClassification(small_config, num_labels=4)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 2], [0, 7, 8, 9, 2]], dtype="int32"))
    attention_mask = ops.convert_to_tensor(np.ones((2, 5), dtype="int32"))

    out = model({"input_ids": input_ids, "attention_mask": attention_mask})
    assert "logits" in out
    assert out["logits"].shape == (2, 4)


def test_esm2_for_token_classification(small_config):
    model = ESM2ForTokenClassification(small_config, num_labels=5)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 2], [0, 7, 8, 9, 2]], dtype="int32"))
    attention_mask = ops.convert_to_tensor(np.ones((2, 5), dtype="int32"))

    out = model({"input_ids": input_ids, "attention_mask": attention_mask})
    assert "logits" in out
    assert out["logits"].shape == (2, 5, 5)


def test_esm2_tokenizer():
    tokenizer = ESMTokenizer()
    seq = "MKTV"
    token_ids = tokenizer.encode(seq)
    # [0, 20, 15, 11, 7, 2]
    assert token_ids[0] == tokenizer.cls_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert len(token_ids) == 6

    # Decode roundtrip
    decoded = tokenizer.decode(token_ids)
    assert decoded == seq

    # Batch encode with padding
    batch = tokenizer(["MKTV", "SEQWENCE"], padding=True, return_tensors="np")
    assert batch["input_ids"].shape == (2, 10)
    assert batch["attention_mask"].shape == (2, 10)
    np.testing.assert_array_equal(batch["attention_mask"][0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(batch["attention_mask"][1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def test_esm2_serialization(small_config):
    model = ESM2Model(small_config)
    input_ids = ops.convert_to_tensor(np.array([[0, 4, 5, 6, 2]], dtype="int32"))
    attention_mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    # Forward pass to build
    orig_out = model(input_ids, attention_mask=attention_mask)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "esm2_model.keras")
        model.save(save_path)

        loaded_model = keras.models.load_model(save_path)
        loaded_out = loaded_model(input_ids, attention_mask=attention_mask)

        np.testing.assert_allclose(
            ops.convert_to_numpy(orig_out["last_hidden_state"]),
            ops.convert_to_numpy(loaded_out["last_hidden_state"]),
            atol=1e-6,
        )

