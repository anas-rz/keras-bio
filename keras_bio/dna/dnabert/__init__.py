"""DNABERT genomic foundation models for Keras 3.

Provides modular Transformer encoders for DNA sequences with k-mer tokenization,
masked language modeling, sequence classification, and zero-dependency k-mer tokenization.
"""

from keras_bio.dna.dnabert.config import DNABERT_PRESETS, DNABERTConfig
from keras_bio.dna.dnabert.convert import (
    convert_dnabert_weights_from_hf,
    load_dnabert_from_hf,
)
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
from keras_bio.dna.dnabert.modeling import (
    DNABERTForMaskedLM,
    DNABERTForSequenceClassification,
    DNABERTForTokenClassification,
    DNABERTModel,
)
from keras_bio.dna.dnabert.tokenizer import DNABERTTokenizer

__all__ = [
    # Config
    "DNABERTConfig",
    "DNABERT_PRESETS",
    # Layers
    "DNABERTEmbeddings",
    "DNABERTSelfAttention",
    "DNABERTAttention",
    "DNABERTIntermediate",
    "DNABERTOutput",
    "DNABERTLayer",
    "DNABERTPooler",
    "DNABERTPredictionHeadTransform",
    "DNABERTLMHead",
    # Models
    "DNABERTModel",
    "DNABERTForMaskedLM",
    "DNABERTForSequenceClassification",
    "DNABERTForTokenClassification",
    # Tokenizer
    "DNABERTTokenizer",
    # Weight Conversion
    "convert_dnabert_weights_from_hf",
    "load_dnabert_from_hf",
]

