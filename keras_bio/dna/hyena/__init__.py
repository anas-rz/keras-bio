"""HyenaDNA genomic foundation models for Keras 3.

Provides sub-quadratic Transformer-alternative architectures for DNA sequences
using implicit long convolutions, gated recurrences, and character-level tokenization.
"""

from keras_bio.dna.hyena.config import HYENADNA_PRESETS, HyenaDNAConfig
from keras_bio.dna.hyena.convert import (
    convert_hyenadna_weights_from_hf,
    load_hyenadna_from_hf,
)
from keras_bio.dna.hyena.layers import (
    HyenaBlock,
    HyenaExponentialModulation,
    HyenaFilter,
    HyenaMlp,
    HyenaOperator,
    HyenaPositionalEmbedding,
    HyenaSin,
    fftconv,
)
from keras_bio.dna.hyena.modeling import (
    HyenaDNAForCausalLM,
    HyenaDNAForSequenceClassification,
    HyenaDNAModel,
)
from keras_bio.dna.hyena.tokenizer import HYENADNA_VOCAB, HyenaDNATokenizer

__all__ = [
    # Config
    "HyenaDNAConfig",
    "HYENADNA_PRESETS",
    # Layers
    "fftconv",
    "HyenaSin",
    "HyenaPositionalEmbedding",
    "HyenaExponentialModulation",
    "HyenaFilter",
    "HyenaOperator",
    "HyenaMlp",
    "HyenaBlock",
    # Models
    "HyenaDNAModel",
    "HyenaDNAForCausalLM",
    "HyenaDNAForSequenceClassification",
    # Tokenizer
    "HyenaDNATokenizer",
    "HYENADNA_VOCAB",
    # Weight Conversion
    "convert_hyenadna_weights_from_hf",
    "load_hyenadna_from_hf",
]

