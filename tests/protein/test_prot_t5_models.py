"""Unit tests for ProtT5 models and tokenization in keras-bio."""

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.protein import (
    ProtT5Config,
    ProtT5EncoderModel,
    ProtT5ForConditionalGeneration,
    ProtT5ForSequenceClassification,
    ProtT5ForTokenClassification,
    ProtT5Model,
    ProtT5Tokenizer,
)


@pytest.fixture
def small_config():
    return ProtT5Config(
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
    )


def test_tokenizer():
    tok = ProtT5Tokenizer()
    seq = "PRTEINO"
    # 'O' should be cleaned to 'X' (token 23)
    encoded = tok.encode(seq, add_special_tokens=True)
    assert encoded[-1] == tok.eos_token_id
    assert encoded[-2] == 23  # 'X'
    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == "PRTEINX"

    # Batch tokenization
    batch = tok(["ACD", "ACDEF"], padding=True, return_tensors="np")
    assert batch["input_ids"].shape == (2, 6)  # longest is 5 + 1 EOS
    assert batch["attention_mask"].shape == (2, 6)
    assert batch["attention_mask"][0, -2] == 0  # padding token mask is 0


def test_encoder_model(small_config):
    model = ProtT5EncoderModel(small_config)
    input_ids = ops.convert_to_tensor([[3, 4, 5, 1], [6, 7, 1, 0]], dtype="int32")
    attention_mask = ops.convert_to_tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype="int32")

    # Dict input
    out_dict = model({"input_ids": input_ids, "attention_mask": attention_mask})
    assert "last_hidden_state" in out_dict
    assert out_dict["last_hidden_state"].shape == (2, 4, 64)

    # Positional input
    out_pos = model(input_ids, attention_mask=attention_mask)
    np.testing.assert_allclose(
        ops.convert_to_numpy(out_dict["last_hidden_state"]),
        ops.convert_to_numpy(out_pos["last_hidden_state"]),
    )

    # embed helper
    embeddings = model.embed(["MKLL", "VLS"], pool="mean")
    assert embeddings.shape == (2, 64)


def test_encoder_decoder_model(small_config):
    model = ProtT5Model(small_config)
    input_ids = ops.convert_to_tensor([[3, 4, 5, 1], [6, 7, 1, 0]], dtype="int32")
    decoder_input_ids = ops.convert_to_tensor([[0, 3, 4], [0, 6, 7]], dtype="int32")

    out = model({"input_ids": input_ids, "decoder_input_ids": decoder_input_ids})
    assert out["last_hidden_state"].shape == (2, 3, 64)
    assert out["encoder_last_hidden_state"].shape == (2, 4, 64)


def test_conditional_generation(small_config):
    model = ProtT5ForConditionalGeneration(small_config)
    input_ids = ops.convert_to_tensor([[3, 4, 5, 1], [6, 7, 1, 0]], dtype="int32")
    out = model({"input_ids": input_ids})
    assert "logits" in out
    assert out["logits"].shape == (2, 4, 128)

    # Test greedy generate
    gen = model.generate(input_ids, max_new_tokens=4)
    assert gen.shape[0] == 2
    assert gen.shape[1] >= 2


def test_sequence_classification(small_config):
    model = ProtT5ForSequenceClassification(small_config, num_classes=5)
    input_ids = ops.convert_to_tensor([[3, 4, 5, 1], [6, 7, 8, 1]], dtype="int32")
    out = model({"input_ids": input_ids})
    assert "logits" in out
    assert out["logits"].shape == (2, 5)


def test_token_classification(small_config):
    model = ProtT5ForTokenClassification(small_config, num_classes=3)
    input_ids = ops.convert_to_tensor([[3, 4, 5, 1], [6, 7, 8, 1]], dtype="int32")
    out = model({"input_ids": input_ids})
    assert "logits" in out
    assert out["logits"].shape == (2, 4, 3)

