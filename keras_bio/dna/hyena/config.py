"""Configuration for HyenaDNA models in Keras 3.

Supports all standard HyenaDNA checkpoints (tiny-1k, tiny-16k, small-32k,
medium-160k, medium-450k, large-1m) with implicit long convolution filters
and character-level nucleotide vocabularies.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


HYENADNA_PRESETS: Dict[str, Dict[str, Any]] = {
    "hyenadna_tiny_1k": {
        "d_model": 128,
        "n_layer": 2,
        "d_inner": 512,
        "max_seq_len": 1026,
    },
    "hyenadna_tiny_1k_d256": {
        "d_model": 256,
        "n_layer": 2,
        "d_inner": 1024,
        "max_seq_len": 1026,
    },
    "hyenadna_tiny_16k_d128": {
        "d_model": 128,
        "n_layer": 2,
        "d_inner": 512,
        "max_seq_len": 16386,
    },
    "hyenadna_small_32k": {
        "d_model": 256,
        "n_layer": 4,
        "d_inner": 1024,
        "max_seq_len": 32770,
    },
    "hyenadna_medium_160k": {
        "d_model": 256,
        "n_layer": 4,
        "d_inner": 1024,
        "max_seq_len": 160002,
    },
    "hyenadna_medium_450k": {
        "d_model": 256,
        "n_layer": 4,
        "d_inner": 1024,
        "max_seq_len": 450002,
    },
    "hyenadna_large_1m": {
        "d_model": 256,
        "n_layer": 8,
        "d_inner": 1024,
        "max_seq_len": 1000002,
    },
}

# Alias mappings for Hugging Face model repository IDs
HYENADNA_PRESETS["hyenadna-tiny-1k-seqlen"] = HYENADNA_PRESETS["hyenadna_tiny_1k"]
HYENADNA_PRESETS["hyenadna-tiny-1k-seqlen-hf"] = HYENADNA_PRESETS["hyenadna_tiny_1k"]
HYENADNA_PRESETS["hyenadna-tiny-1k-seqlen-d256"] = HYENADNA_PRESETS["hyenadna_tiny_1k_d256"]
HYENADNA_PRESETS["hyenadna-tiny-1k-seqlen-d256-hf"] = HYENADNA_PRESETS["hyenadna_tiny_1k_d256"]
HYENADNA_PRESETS["hyenadna-tiny-16k-seqlen-d128"] = HYENADNA_PRESETS["hyenadna_tiny_16k_d128"]
HYENADNA_PRESETS["hyenadna-tiny-16k-seqlen-d128-hf"] = HYENADNA_PRESETS["hyenadna_tiny_16k_d128"]
HYENADNA_PRESETS["hyenadna-small-32k-seqlen"] = HYENADNA_PRESETS["hyenadna_small_32k"]
HYENADNA_PRESETS["hyenadna-small-32k-seqlen-hf"] = HYENADNA_PRESETS["hyenadna_small_32k"]
HYENADNA_PRESETS["hyenadna-medium-160k-seqlen"] = HYENADNA_PRESETS["hyenadna_medium_160k"]
HYENADNA_PRESETS["hyenadna-medium-160k-seqlen-hf"] = HYENADNA_PRESETS["hyenadna_medium_160k"]
HYENADNA_PRESETS["hyenadna-medium-450k-seqlen"] = HYENADNA_PRESETS["hyenadna_medium_450k"]
HYENADNA_PRESETS["hyenadna-medium-450k-seqlen-hf"] = HYENADNA_PRESETS["hyenadna_medium_450k"]
HYENADNA_PRESETS["hyenadna-large-1m-seqlen"] = HYENADNA_PRESETS["hyenadna_large_1m"]
HYENADNA_PRESETS["hyenadna-large-1m-seqlen-hf"] = HYENADNA_PRESETS["hyenadna_large_1m"]


@dataclass
class HyenaDNAConfig:
    """Architecture configuration for HyenaDNA models.

    Args:
        d_model: Dimensionality of the model and embeddings.
        n_layer: Number of Hyena blocks.
        d_inner: Dimensionality of the inner feed-forward MLP.
        vocab_size: Unpadded vocabulary size (typically 12).
        pad_vocab_size_multiple: Multiplier for padding vocabulary size (default 8).
        max_seq_len: Maximum supported sequence length for filters.
        filter_order: Dimensionality of the implicit filter MLP width.
        short_filter_order: Kernel size of the causal depthwise short convolution.
        emb_dim: Dimension of positional encoding fed to implicit filter.
        hyena_order: Order of the Hyena recurrence (depth, default 2).
        activation_freq: Frequency multiplier for sine activation function.
        num_inner_mlps: Number of intermediate layers in the implicit filter MLP.
        layer_norm_epsilon: Epsilon for LayerNormalization layers.
        embed_dropout: Dropout probability on input embeddings.
        hyena_dropout: Dropout probability inside the Hyena operator recurrence.
        hyena_filter_dropout: Dropout probability for the implicit filter.
        use_bias: Whether to use bias in linear and filter projections.
        train_freq: Whether the sine frequency parameter is trainable.
        tie_word_embeddings: Whether to tie LM head weights to word embeddings.
        pad_token_id: The ID of the [PAD] token.
        num_labels: Number of labels for sequence classification head.
    """

    d_model: int = 128
    n_layer: int = 2
    d_inner: int = 512
    vocab_size: int = 12
    pad_vocab_size_multiple: int = 8
    max_seq_len: int = 1026
    filter_order: int = 64
    short_filter_order: int = 3
    emb_dim: int = 5
    hyena_order: int = 2
    activation_freq: float = 10.0
    num_inner_mlps: int = 2
    layer_norm_epsilon: float = 1e-5
    embed_dropout: float = 0.1
    hyena_dropout: float = 0.0
    hyena_filter_dropout: float = 0.0
    use_bias: bool = True
    train_freq: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: int = 4
    num_labels: int = 2

    @property
    def padded_vocab_size(self) -> int:
        """Returns vocabulary size rounded up to the nearest multiple."""
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            return self.vocab_size + self.pad_vocab_size_multiple - (
                self.vocab_size % self.pad_vocab_size_multiple
            )
        return self.vocab_size

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["padded_vocab_size"] = self.padded_vocab_size
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyenaDNAConfig":
        valid = {f.name for f in fields(cls)}
        mapped = dict(d)
        if "padded_vocab_size" in mapped and "vocab_size" not in mapped:
            mapped["vocab_size"] = mapped.pop("padded_vocab_size")
        filtered = {k: v for k, v in mapped.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs: Any) -> "HyenaDNAConfig":
        """Instantiates HyenaDNAConfig from a known checkpoint preset name."""
        clean_name = preset_name.replace("LongSafari/", "")
        if clean_name not in HYENADNA_PRESETS:
            valid_names = ", ".join(HYENADNA_PRESETS.keys())
            raise ValueError(
                f"Unknown HyenaDNA preset '{preset_name}'. Valid presets are: {valid_names}"
            )
        preset_kwargs = dict(HYENADNA_PRESETS[clean_name])
        preset_kwargs.update(kwargs)
        return cls(**preset_kwargs)

    @classmethod
    def from_hf(cls, hf_config: Dict[str, Any]) -> "HyenaDNAConfig":
        """Maps a Hugging Face HyenaDNA config dictionary to HyenaDNAConfig."""
        return cls(
            d_model=hf_config.get("d_model", 128),
            n_layer=hf_config.get("n_layer", 2),
            d_inner=hf_config.get("d_inner", 512),
            vocab_size=hf_config.get("vocab_size", 12),
            pad_vocab_size_multiple=hf_config.get("pad_vocab_size_multiple", 8),
            max_seq_len=hf_config.get("max_seq_len", 1026),
            filter_order=hf_config.get("filter_order", 64),
            short_filter_order=hf_config.get("short_filter_order", 3),
            emb_dim=hf_config.get("emb_dim", 5),
            hyena_order=hf_config.get("hyena_order", 2),
            activation_freq=float(hf_config.get("activation_freq", 10.0)),
            num_inner_mlps=hf_config.get("num_inner_mlps", 2),
            layer_norm_epsilon=float(hf_config.get("layer_norm_epsilon", 1e-5)),
            embed_dropout=float(hf_config.get("embed_dropout", 0.1)),
            hyena_dropout=float(hf_config.get("hyena_dropout", 0.0)),
            hyena_filter_dropout=float(hf_config.get("hyena_filter_dropout", 0.0)),
            use_bias=bool(hf_config.get("use_bias", True)),
            train_freq=bool(hf_config.get("train_freq", True)),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
            pad_token_id=hf_config.get("pad_token_id", 4),
            num_labels=hf_config.get("num_labels", 2),
        )

