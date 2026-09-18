"""Unit tests for DNABERT models and tokenizer in keras-bio."""

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.dna.dnabert.config import DNABERTConfig
from keras_bio.dna.dnabert.modeling import (
    DNABERTForMaskedLM,
    DNABERTForSequenceClassification,
    DNABERTForTokenClassification,
    DNABERTModel,
)
from keras_bio.dna.dnabert.tokenizer import DNABERTTokenizer


@pytest.fixture
def dnabert_config():
    return DNABERTConfig(
        kmer=6,
        vocab_size=4101,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def test_dnabert_tokenizer():
    # Test different k-mer sizes
    for k, expected_vocab in [(3, 69), (4, 261), (5, 1029), (6, 4101)]:
        tok = DNABERTTokenizer(kmer=k)
        assert tok.vocab_size == expected_vocab
        assert tok.cls_token_id == 2
        assert tok.sep_token_id == 3
        assert tok.pad_token_id == 0

    tok = DNABERTTokenizer(kmer=6)
    dna = "ATCGATCGATCGATCG"
    kmers = tok.dna_to_kmers(dna)
    assert len(kmers) == len(dna) - 6 + 1
    reconstructed = tok.kmers_to_dna(kmers)
    assert reconstructed == dna

    # Test single sequence encode
    ids = tok.encode(dna, add_special_tokens=True)
    assert ids[0] == tok.cls_token_id
    assert ids[-1] == tok.sep_token_id
    assert len(ids) == len(kmers) + 2

    # Test batch encode with padding
    batch_enc = tok([dna, "ATCGATCG"], padding=True, return_tensors="np")
    assert batch_enc["input_ids"].shape[0] == 2
    assert batch_enc["attention_mask"].shape[0] == 2
    assert batch_enc["input_ids"].shape[1] == len(kmers) + 2

    # Test decode
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert decoded == " ".join(kmers)


def test_dnabert_model_forward(dnabert_config):
    model = DNABERTModel(dnabert_config)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 20, 30, 3], [2, 5, 6, 7, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((2, 5), dtype="int32"))

    # Basic forward pass
    outputs = model({"input_ids": input_ids, "attention_mask": mask})
    assert "last_hidden_state" in outputs
    assert "pooler_output" in outputs
    assert outputs["last_hidden_state"].shape == (2, 5, 64)
    assert outputs["pooler_output"].shape == (2, 64)

    # Output hidden states and attentions
    full_outputs = model(
        {"input_ids": input_ids, "attention_mask": mask},
        output_hidden_states=True,
        output_attentions=True,
    )
    assert "hidden_states" in full_outputs
    assert len(full_outputs["hidden_states"]) == dnabert_config.num_hidden_layers + 1
    assert "attentions" in full_outputs
    assert len(full_outputs["attentions"]) == dnabert_config.num_hidden_layers


def test_dnabert_embed_api(dnabert_config):
    model = DNABERTModel(dnabert_config)
    tok = DNABERTTokenizer(kmer=6)
    seqs = ["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]

    # Mean pooling
    emb_mean = model.embed(seqs, tokenizer=tok, pool="mean")
    assert isinstance(emb_mean, np.ndarray)
    assert emb_mean.shape == (2, 64)
    assert not np.isnan(emb_mean).any()

    # CLS pooling
    emb_cls = model.embed(seqs, tokenizer=tok, pool="cls")
    assert isinstance(emb_cls, np.ndarray)
    assert emb_cls.shape == (2, 64)
    assert not np.isnan(emb_cls).any()

    # Per-kmer pooling (None)
    emb_none = model.embed(seqs, tokenizer=tok, pool=None)
    assert isinstance(emb_none, np.ndarray)
    assert emb_none.shape[0] == 2
    assert emb_none.shape[2] == 64

    # Single string input and return_numpy=False
    emb_tensor = model.embed("ATCGATCGATCG", tokenizer=tok, pool="mean", return_numpy=False)
    assert emb_tensor.shape == (1, 64)


def test_dnabert_for_masked_lm(dnabert_config):
    model = DNABERTForMaskedLM(dnabert_config)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 4, 30, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    outputs = model({"input_ids": input_ids, "attention_mask": mask})
    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 5, 4101)


def test_dnabert_for_sequence_classification(dnabert_config):
    model = DNABERTForSequenceClassification(dnabert_config, num_labels=3)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 20, 30, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    outputs = model({"input_ids": input_ids, "attention_mask": mask})
    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 3)
    assert "pooler_output" in outputs
    assert outputs["pooler_output"].shape == (1, 64)


def test_dnabert_for_token_classification(dnabert_config):
    model = DNABERTForTokenClassification(dnabert_config, num_labels=5)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 20, 30, 3]], dtype="int32"))
    mask = ops.convert_to_tensor(np.ones((1, 5), dtype="int32"))

    outputs = model({"input_ids": input_ids, "attention_mask": mask})
    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 5, 5)


def test_dnabert_presets():
    for name, expected_kmer, expected_vocab in [
        ("dnabert_3", 3, 69),
        ("dnabert_4", 4, 261),
        ("dnabert_5", 5, 1029),
        ("dnabert_6", 6, 4101),
        ("DNA_bert_6", 6, 4101),
    ]:
        cfg = DNABERTConfig.from_preset(name)
        assert cfg.kmer == expected_kmer
        assert cfg.vocab_size == expected_vocab
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12

