"""Unit tests for DNABERT layers in keras-bio."""

import keras
from keras import ops
import numpy as np
import pytest

from keras_bio.dna.dnabert.config import DNABERTConfig
from keras_bio.dna.dnabert.layers import (
    DNABERTAttention,
    DNABERTEmbeddings,
    DNABERTIntermediate,
    DNABERTLayer,
    DNABERTLMHead,
    DNABERTOutput,
    DNABERTPooler,
    DNABERTPredictionHeadTransform,
    DNABERTSelfAttention,
)


@pytest.fixture
def dnabert_config():
    return DNABERTConfig(
        vocab_size=69,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def test_dnabert_embeddings(dnabert_config):
    emb_layer = DNABERTEmbeddings(dnabert_config)
    input_ids = ops.convert_to_tensor(np.array([[2, 10, 15, 3]], dtype="int32"))
    token_types = ops.convert_to_tensor(np.array([[0, 0, 0, 0]], dtype="int32"))
    pos_ids = ops.convert_to_tensor(np.array([[0, 1, 2, 3]], dtype="int32"))

    out = emb_layer(input_ids, token_type_ids=token_types, position_ids=pos_ids)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == (1, 4, 64)
    assert not np.isnan(out_np).any()

    # Default position_ids and token_type_ids
    out_default = emb_layer(input_ids)
    assert out_default.shape == (1, 4, 64)

    # Serialization
    config = emb_layer.get_config()
    reconstructed = DNABERTEmbeddings.from_config(config)
    assert reconstructed.config.vocab_size == 69
    assert reconstructed.config.hidden_size == 64


def test_dnabert_self_attention(dnabert_config):
    attn = DNABERTSelfAttention(dnabert_config)
    x = ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32"))
    mask = ops.convert_to_tensor(np.ones((2, 6), dtype="int32"))

    out = attn(x, attention_mask=mask)
    assert out.shape == (2, 6, 64)

    out, weights = attn(x, attention_mask=mask, output_attentions=True)
    assert out.shape == (2, 6, 64)
    assert weights.shape == (2, 4, 6, 6)

    # Serialization
    config = attn.get_config()
    reconstructed = DNABERTSelfAttention.from_config(config)
    assert reconstructed.hidden_size == 64
    assert reconstructed.num_attention_heads == 4


def test_dnabert_attention(dnabert_config):
    attn_sublayer = DNABERTAttention(dnabert_config)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))

    out = attn_sublayer(x)
    assert out.shape == (2, 5, 64)

    out, weights = attn_sublayer(x, output_attentions=True)
    assert out.shape == (2, 5, 64)
    assert weights.shape == (2, 4, 5, 5)

    # Serialization
    config = attn_sublayer.get_config()
    reconstructed = DNABERTAttention.from_config(config)
    assert reconstructed.config.hidden_size == 64


def test_dnabert_intermediate(dnabert_config):
    inter = DNABERTIntermediate(dnabert_config)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))

    out = inter(x)
    assert out.shape == (2, 5, 128)

    # Serialization
    config = inter.get_config()
    reconstructed = DNABERTIntermediate.from_config(config)
    assert reconstructed.config.intermediate_size == 128


def test_dnabert_output(dnabert_config):
    out_layer = DNABERTOutput(dnabert_config)
    intermediate = ops.convert_to_tensor(np.random.randn(2, 5, 128).astype("float32"))
    residual = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))

    out = out_layer(intermediate, residual)
    assert out.shape == (2, 5, 64)

    # Serialization
    config = out_layer.get_config()
    reconstructed = DNABERTOutput.from_config(config)
    assert reconstructed.config.hidden_size == 64


def test_dnabert_layer(dnabert_config):
    encoder_block = DNABERTLayer(dnabert_config)
    x = ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32"))

    out = encoder_block(x)
    assert out.shape == (2, 6, 64)

    out, weights = encoder_block(x, output_attentions=True)
    assert out.shape == (2, 6, 64)
    assert weights.shape == (2, 4, 6, 6)

    # Serialization
    config = encoder_block.get_config()
    reconstructed = DNABERTLayer.from_config(config)
    assert reconstructed.config.hidden_size == 64


def test_dnabert_pooler(dnabert_config):
    pooler = DNABERTPooler(dnabert_config)
    x = ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32"))

    out = pooler(x)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == (2, 64)
    # Tanh bounds
    assert np.all(out_np >= -1.0)
    assert np.all(out_np <= 1.0)

    # Serialization
    config = pooler.get_config()
    reconstructed = DNABERTPooler.from_config(config)
    assert reconstructed.config.hidden_size == 64


def test_dnabert_prediction_head_transform(dnabert_config):
    transform = DNABERTPredictionHeadTransform(dnabert_config)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))

    out = transform(x)
    assert out.shape == (2, 5, 64)

    # Serialization
    config = transform.get_config()
    reconstructed = DNABERTPredictionHeadTransform.from_config(config)
    assert reconstructed.config.hidden_size == 64


def test_dnabert_lm_head(dnabert_config):
    tied_weights = ops.convert_to_tensor(np.random.randn(69, 64).astype("float32"))
    lm_head = DNABERTLMHead(dnabert_config, tied_embedding_weights=tied_weights)
    x = ops.convert_to_tensor(np.random.randn(2, 5, 64).astype("float32"))

    out = lm_head(x)
    assert out.shape == (2, 5, 69)

    # Untied test
    untied_cfg = DNABERTConfig(
        vocab_size=69,
        hidden_size=64,
        tie_word_embeddings=False,
    )
    untied_head = DNABERTLMHead(untied_cfg)
    out_untied = untied_head(x)
    assert out_untied.shape == (2, 5, 69)

    # Serialization
    config = lm_head.get_config()
    reconstructed = DNABERTLMHead.from_config(config)
    assert reconstructed.config.vocab_size == 69

