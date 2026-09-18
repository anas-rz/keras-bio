"""Unit tests for HyenaDNA models and tokenizer in keras-bio."""

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.dna.hyena.config import HyenaDNAConfig
from keras_bio.dna.hyena.modeling import (
    HyenaDNAForCausalLM,
    HyenaDNAForSequenceClassification,
    HyenaDNAModel,
)
from keras_bio.dna.hyena.tokenizer import HyenaDNATokenizer


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


def test_hyenadna_tokenizer():
    tok = HyenaDNATokenizer()
    assert tok.vocab_size == 12
    assert tok.sep_token_id == 1
    assert tok.pad_token_id == 4

    seq = "ATCGATCG"
    enc = tok.encode(seq)
    assert enc[-1] == tok.sep_token_id
    assert tok.decode(enc) == seq

    batch = tok(["ATCG", "ATCGATCG"], padding=True, return_tensors="np")
    assert batch["input_ids"].shape == (2, 9)
    assert batch["attention_mask"].shape == (2, 9)


def test_hyenadna_model_forward(hyena_config):
    model = HyenaDNAModel(hyena_config)
    input_ids = ops.convert_to_tensor(np.array([[7, 8, 9, 10, 1], [7, 8, 7, 8, 1]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((2, 5), dtype="int32"))

    outputs = model({"input_ids": input_ids, "attention_mask": mask})
    assert "last_hidden_state" in outputs
    assert outputs["last_hidden_state"].shape == (2, 5, 64)

    full_outputs = model(
        {"input_ids": input_ids, "attention_mask": mask},
        output_hidden_states=True,
    )
    assert "hidden_states" in full_outputs
    assert len(full_outputs["hidden_states"]) == hyena_config.n_layer + 1


def test_hyenadna_embed_api(hyena_config):
    model = HyenaDNAModel(hyena_config)
    tok = HyenaDNATokenizer()
    seqs = ["ATCGATCG", "GCTAGCTA"]

    emb_mean = model.embed(seqs, tokenizer=tok, pool="mean")
    assert isinstance(emb_mean, np.ndarray)
    assert emb_mean.shape == (2, 64)
    assert not np.isnan(emb_mean).any()

    emb_last = model.embed(seqs, tokenizer=tok, pool="last")
    assert isinstance(emb_last, np.ndarray)
    assert emb_last.shape == (2, 64)
    assert not np.isnan(emb_last).any()

    emb_none = model.embed(seqs, tokenizer=tok, pool=None)
    assert isinstance(emb_none, np.ndarray)
    assert emb_none.shape[0] == 2
    assert emb_none.shape[2] == 64

    # Single string input and return_numpy=False
    emb_tensor = model.embed("ATCG", tokenizer=tok, pool="mean", return_numpy=False)
    assert emb_tensor.shape == (1, 64)


def test_hyenadna_causal_lm(hyena_config):
    model = HyenaDNAForCausalLM(hyena_config)
    input_ids = ops.convert_to_tensor(np.array([[7, 8, 9, 10, 1]], dtype="int32"))

    outputs = model(input_ids)
    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 5, 16)


def test_hyenadna_sequence_classification(hyena_config):
    model = HyenaDNAForSequenceClassification(hyena_config, num_labels=3)
    input_ids = ops.convert_to_tensor(np.array([[7, 8, 9, 10, 1], [7, 8, 4, 4, 4]], dtype="int32"))

    outputs = model(input_ids)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 3)
    assert "pooled_hidden_state" in outputs
    assert outputs["pooled_hidden_state"].shape == (2, 64)


def test_hyenadna_presets():
    for name, expected_d_model, expected_layers, expected_len in [
        ("hyenadna_tiny_1k", 128, 2, 1026),
        ("hyenadna_tiny_1k_d256", 256, 2, 1026),
        ("hyenadna_tiny_16k_d128", 128, 2, 16386),
        ("hyenadna_small_32k", 256, 4, 32770),
        ("hyenadna_medium_160k", 256, 4, 160002),
        ("hyenadna_large_1m", 256, 8, 1000002),
    ]:
        cfg = HyenaDNAConfig.from_preset(name)
        assert cfg.d_model == expected_d_model
        assert cfg.n_layer == expected_layers
        assert cfg.max_seq_len == expected_len

