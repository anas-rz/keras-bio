"""Multi-backend execution and serialization tests for DNABERT."""

import os
import tempfile

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.dna.dnabert import (
    DNABERTConfig,
    DNABERTForMaskedLM,
    DNABERTForSequenceClassification,
    DNABERTForTokenClassification,
    DNABERTModel,
    DNABERTTokenizer,
)


@pytest.fixture
def dnabert_config():
    return DNABERTConfig(
        kmer=6,
        vocab_size=4101,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def test_multibackend_forward(dnabert_config):
    """Test DNABERTModel forward pass on current backend."""
    model = DNABERTModel(dnabert_config)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 20, 30, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    out = model(input_ids, attention_mask=mask)
    assert "last_hidden_state" in out
    hidden = ops.convert_to_numpy(out["last_hidden_state"])
    assert hidden.shape == (1, 5, 64)
    assert not np.isnan(hidden).any()

    assert "pooler_output" in out
    pooler = ops.convert_to_numpy(out["pooler_output"])
    assert pooler.shape == (1, 64)
    assert not np.isnan(pooler).any()


def test_multibackend_masked_lm(dnabert_config):
    """Test DNABERTForMaskedLM forward pass on current backend."""
    model = DNABERTForMaskedLM(dnabert_config)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 4, 30, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    out = model(input_ids, attention_mask=mask)
    assert "logits" in out
    logits = ops.convert_to_numpy(out["logits"])
    assert logits.shape == (1, 5, 4101)
    assert not np.isnan(logits).any()


def test_multibackend_classification_heads(dnabert_config):
    """Test sequence and token classification heads."""
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 20, 30, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    seq_clf = DNABERTForSequenceClassification(dnabert_config, num_labels=2)
    seq_out = seq_clf(input_ids, attention_mask=mask)
    seq_logits = ops.convert_to_numpy(seq_out["logits"])
    assert seq_logits.shape == (1, 2)
    assert not np.isnan(seq_logits).any()

    tok_clf = DNABERTForTokenClassification(dnabert_config, num_labels=4)
    tok_out = tok_clf(input_ids, attention_mask=mask)
    tok_logits = ops.convert_to_numpy(tok_out["logits"])
    assert tok_logits.shape == (1, 5, 4)
    assert not np.isnan(tok_logits).any()


def test_multibackend_embed_api(dnabert_config):
    """Test embed() method with DNA sequence strings."""
    model = DNABERTModel(dnabert_config)
    tokenizer = DNABERTTokenizer(kmer=6)
    seqs = ["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]

    pooled = model.embed(seqs, tokenizer=tokenizer, pool="mean")
    assert isinstance(pooled, np.ndarray)
    assert pooled.shape == (2, 64)
    assert not np.isnan(pooled).any()

    cls_pooled = model.embed(seqs, tokenizer=tokenizer, pool="cls")
    assert isinstance(cls_pooled, np.ndarray)
    assert cls_pooled.shape == (2, 64)
    assert not np.isnan(cls_pooled).any()


def test_multibackend_save_load_roundtrip(dnabert_config):
    """Verify DNABERT model saves to .keras and restores exact weights and outputs."""
    model = DNABERTModel(dnabert_config)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 20, 30, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    out_before = ops.convert_to_numpy(model(input_ids, attention_mask=mask)["last_hidden_state"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_dnabert.keras")
        model.save(path)

        loaded = keras.models.load_model(path)
        out_after = ops.convert_to_numpy(loaded(input_ids, attention_mask=mask)["last_hidden_state"])

        np.testing.assert_allclose(out_before, out_after, atol=1e-6)

