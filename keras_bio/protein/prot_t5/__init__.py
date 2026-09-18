"""ProtT5 (Protein T5) Foundation Model in Keras 3.

Provides multi-backend implementations for protein sequence representation,
generation, and classification.
"""

from keras_bio.protein.prot_t5.config import ProtT5Config, T5Config
from keras_bio.protein.prot_t5.convert import (
    WeightMappingError,
    load_t5_encoder_from_hf,
    load_t5_model_from_hf,
    transfer_t5_weights,
    transfer_weights,
)
from keras_bio.protein.prot_t5.layers import (
    ProtT5Attention,
    ProtT5CrossAttentionLayer,
    ProtT5DecoderBlock,
    ProtT5DenseActDense,
    ProtT5DenseGatedDense,
    ProtT5EncoderBlock,
    ProtT5FeedForwardLayer,
    ProtT5LayerNorm,
    ProtT5PositionBias,
    ProtT5SelfAttentionLayer,
    ProtT5TiedHead,
    T5Attention,
    T5CrossAttentionLayer,
    T5DecoderBlock,
    T5DenseActDense,
    T5DenseGatedDense,
    T5EncoderBlock,
    T5FeedForwardLayer,
    T5LayerNorm,
    T5PositionBias,
    T5SelfAttentionLayer,
    T5TiedHead,
)
from keras_bio.protein.prot_t5.modeling import (
    ProtT5EncoderModel,
    ProtT5ForConditionalGeneration,
    ProtT5ForSequenceClassification,
    ProtT5ForTokenClassification,
    ProtT5Model,
    T5ConditionalGenerate,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5Model,
    T5SequenceClassify,
    T5TokenClassify,
)
from keras_bio.protein.prot_t5.tokenizer import (
    ProtT5Tokenizer,
    clean_protein_sequence,
    format_for_t5,
)

__all__ = [
    # Config
    "ProtT5Config",
    "T5Config",
    # Layers
    "ProtT5LayerNorm",
    "ProtT5PositionBias",
    "ProtT5Attention",
    "ProtT5DenseActDense",
    "ProtT5DenseGatedDense",
    "ProtT5FeedForwardLayer",
    "ProtT5SelfAttentionLayer",
    "ProtT5CrossAttentionLayer",
    "ProtT5EncoderBlock",
    "ProtT5DecoderBlock",
    "ProtT5TiedHead",
    "T5LayerNorm",
    "T5PositionBias",
    "T5Attention",
    "T5DenseActDense",
    "T5DenseGatedDense",
    "T5FeedForwardLayer",
    "T5SelfAttentionLayer",
    "T5CrossAttentionLayer",
    "T5EncoderBlock",
    "T5DecoderBlock",
    "T5TiedHead",
    # Models
    "ProtT5EncoderModel",
    "ProtT5Model",
    "ProtT5ForConditionalGeneration",
    "ProtT5ForSequenceClassification",
    "ProtT5ForTokenClassification",
    "T5EncoderModel",
    "T5Model",
    "T5ConditionalGenerate",
    "T5ForConditionalGeneration",
    "T5SequenceClassify",
    "T5TokenClassify",
    # Conversion
    "WeightMappingError",
    "transfer_t5_weights",
    "transfer_weights",
    "load_t5_encoder_from_hf",
    "load_t5_model_from_hf",
    # Tokenization & Utilities
    "ProtT5Tokenizer",
    "clean_protein_sequence",
    "format_for_t5",
]

