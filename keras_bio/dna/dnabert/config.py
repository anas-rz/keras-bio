"""Configuration for DNABERT models in Keras 3.

Supports all standard DNABERT checkpoints (DNA_bert_3, DNA_bert_4,
DNA_bert_5, DNA_bert_6) with k-mer vocabularies and standard BERT encoder architectures.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


DNABERT_PRESETS: Dict[str, Dict[str, Any]] = {
    "dnabert_3": {
        "kmer": 3,
        "vocab_size": 69,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "dnabert_4": {
        "kmer": 4,
        "vocab_size": 261,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "dnabert_5": {
        "kmer": 5,
        "vocab_size": 1029,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "dnabert_6": {
        "kmer": 6,
        "vocab_size": 4101,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
}

# Add alias mappings for Hugging Face model IDs
DNABERT_PRESETS["DNA_bert_3"] = DNABERT_PRESETS["dnabert_3"]
DNABERT_PRESETS["DNA_bert_4"] = DNABERT_PRESETS["dnabert_4"]
DNABERT_PRESETS["DNA_bert_5"] = DNABERT_PRESETS["dnabert_5"]
DNABERT_PRESETS["DNA_bert_6"] = DNABERT_PRESETS["dnabert_6"]


@dataclass
class DNABERTConfig:
    """Architecture configuration for DNABERT models.

    Args:
        kmer: Length of k-mers used by the tokenizer (3, 4, 5, or 6).
        vocab_size: Vocabulary size (default 4101 for 6-mer DNABERT).
        hidden_size: Dimensionality of the encoder layers and embeddings.
        num_hidden_layers: Number of Transformer encoder layers.
        num_attention_heads: Number of attention heads in each attention layer.
        intermediate_size: Dimensionality of the intermediate feed-forward layer.
        hidden_act: Non-linear activation function in the encoder and pooler.
        hidden_dropout_prob: Dropout probability for all fully connected layers.
        attention_probs_dropout_prob: Dropout ratio for the attention probabilities.
        max_position_embeddings: Maximum sequence length the model might ever use.
        type_vocab_size: Vocabulary size for `token_type_ids`.
        initializer_range: Standard deviation of truncated normal weight initializer.
        layer_norm_eps: Epsilon used by LayerNormalization.
        pad_token_id: The ID of the [PAD] token.
        unk_token_id: The ID of the [UNK] token.
        cls_token_id: The ID of the [CLS] token.
        sep_token_id: The ID of the [SEP] token.
        mask_token_id: The ID of the [MASK] token.
        num_labels: Number of labels for sequence or token classification heads.
        classifier_dropout: Dropout ratio for classification heads.
        tie_word_embeddings: Whether to tie input word embeddings and LM head projection.
    """

    kmer: int = 6
    vocab_size: int = 4101
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4
    num_labels: int = 2
    classifier_dropout: Optional[float] = None
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def embed_dim(self) -> int:
        return self.hidden_size

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @property
    def mlp_dim(self) -> int:
        return self.intermediate_size

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["head_dim"] = self.head_dim
        d["embed_dim"] = self.embed_dim
        d["num_layers"] = self.num_layers
        d["num_heads"] = self.num_heads
        d["mlp_dim"] = self.mlp_dim
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DNABERTConfig":
        valid = {f.name for f in fields(cls)}
        mapped = dict(d)
        if "embed_dim" in mapped and "hidden_size" not in mapped:
            mapped["hidden_size"] = mapped.pop("embed_dim")
        if "num_layers" in mapped and "num_hidden_layers" not in mapped:
            mapped["num_hidden_layers"] = mapped.pop("num_layers")
        if "num_heads" in mapped and "num_attention_heads" not in mapped:
            mapped["num_attention_heads"] = mapped.pop("num_heads")
        if "mlp_dim" in mapped and "intermediate_size" not in mapped:
            mapped["intermediate_size"] = mapped.pop("mlp_dim")
        filtered = {k: v for k, v in mapped.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs: Any) -> "DNABERTConfig":
        """Instantiates a DNABERTConfig from a known checkpoint preset name."""
        clean_name = preset_name.replace("zhihan1996/", "")
        if clean_name not in DNABERT_PRESETS:
            valid_names = ", ".join(DNABERT_PRESETS.keys())
            raise ValueError(
                f"Unknown DNABERT preset '{preset_name}'. Valid presets are: {valid_names}"
            )
        preset_kwargs = dict(DNABERT_PRESETS[clean_name])
        preset_kwargs.update(kwargs)
        return cls(**preset_kwargs)

    @classmethod
    def from_hf(cls, hf_config: Dict[str, Any]) -> "DNABERTConfig":
        """Maps a Hugging Face BERT/DNABERT config dictionary to DNABERTConfig."""
        vocab_size = hf_config.get("vocab_size", 4101)
        # Infer kmer from vocab size if possible: 4^k + 5
        kmer_guess = 6
        for k in (3, 4, 5, 6):
            if vocab_size == 4**k + 5:
                kmer_guess = k
                break

        return cls(
            kmer=hf_config.get("kmer", kmer_guess),
            vocab_size=vocab_size,
            hidden_size=hf_config.get("hidden_size", 768),
            num_hidden_layers=hf_config.get("num_hidden_layers", 12),
            num_attention_heads=hf_config.get("num_attention_heads", 12),
            intermediate_size=hf_config.get("intermediate_size", 3072),
            hidden_act=hf_config.get("hidden_act", "gelu"),
            hidden_dropout_prob=hf_config.get("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob=hf_config.get("attention_probs_dropout_prob", 0.1),
            max_position_embeddings=hf_config.get("max_position_embeddings", 512),
            type_vocab_size=hf_config.get("type_vocab_size", 2),
            initializer_range=hf_config.get("initializer_range", 0.02),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-12),
            pad_token_id=hf_config.get("pad_token_id", 0),
            unk_token_id=hf_config.get("unk_token_id", 1),
            cls_token_id=hf_config.get("cls_token_id", 2),
            sep_token_id=hf_config.get("sep_token_id", 3),
            mask_token_id=hf_config.get("mask_token_id", 4),
            num_labels=hf_config.get("num_labels", 2),
            classifier_dropout=hf_config.get("classifier_dropout", None),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
        )

