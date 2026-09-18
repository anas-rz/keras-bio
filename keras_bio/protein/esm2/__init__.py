"""ESM-2 (Evolutionary Scale Modeling) protein language models for Keras 3.

Provides modular Transformer encoders with Rotary Position Embeddings (RoPE),
contact prediction with Average Product Correction (APC), masked language modeling,
sequence classification, and zero-dependency protein tokenization.
"""

from keras_bio.protein.esm2.config import ESM2_PRESETS, ESM2Config
from keras_bio.protein.esm2.convert import convert_esm2_weights_from_hf, load_esm2_from_hf
from keras_bio.protein.esm2.layers import (
    ESM2Attention,
    ESM2ClassificationHead,
    ESM2ContactPredictionHead,
    ESM2EncoderLayer,
    ESM2FeedForward,
    ESM2LMHead,
    ESM2RotaryEmbedding,
    ESM2SelfAttention,
    apply_rotary_pos_emb,
    average_product_correct,
    rotate_half,
    symmetrize,
)
from keras_bio.protein.esm2.modeling import (
    ESM2ForMaskedLM,
    ESM2ForSequenceClassification,
    ESM2ForTokenClassification,
    ESM2Model,
)
from keras_bio.protein.esm2.tokenizer import ESM_VOCAB, ESMTokenizer

__all__ = [
    # Config
    "ESM2Config",
    "ESM2_PRESETS",
    # Layers
    "rotate_half",
    "apply_rotary_pos_emb",
    "symmetrize",
    "average_product_correct",
    "ESM2RotaryEmbedding",
    "ESM2SelfAttention",
    "ESM2Attention",
    "ESM2FeedForward",
    "ESM2EncoderLayer",
    "ESM2ContactPredictionHead",
    "ESM2LMHead",
    "ESM2ClassificationHead",
    # Models
    "ESM2Model",
    "ESM2ForMaskedLM",
    "ESM2ForSequenceClassification",
    "ESM2ForTokenClassification",
    # Tokenizer
    "ESMTokenizer",
    "ESM_VOCAB",
    # Weight Conversion
    "convert_esm2_weights_from_hf",
    "load_esm2_from_hf",
]

