"""Configuration for ProtT5 models in Keras 3.

Supports both original ungated T5 (ProtT5-XL) and gated-activation variants
(ProtT5-base-mt, T5-v1.1, mT5).
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class ProtT5Config:
    """Architecture configuration for ProtT5 and T5 models.

    Args:
        vocab_size: Number of tokens in vocabulary. ProtT5-XL has 128 tokens;
            ProtT5-base has 256 tokens; standard NLP T5 has 32,128 tokens.
        d_model: Dimensionality of the encoder/decoder layers (embed dimension).
        d_kv: Dimensionality of key and value projections per head.
        d_ff: Dimensionality of the feed-forward intermediate projections.
        num_layers: Number of hidden layers in the Transformer encoder.
        num_decoder_layers: Number of hidden layers in the Transformer decoder.
            If not specified, defaults to `num_layers`.
        num_heads: Number of attention heads for each attention layer.
        relative_attention_num_buckets: Number of buckets for relative position bias.
        relative_attention_max_distance: Maximum distance for relative position bias.
        dropout_rate: Float between 0 and 1 for dropout probabilities.
        layer_norm_epsilon: Epsilon for layer normalization / RMSNorm.
        initializer_factor: Factor for initializing model weights.
        feed_forward_proj: Type of feed-forward layer projection ("relu", "gated-gelu", etc.).
        dense_act_fn: Activation function used in feed-forward layers.
        is_gated_act: Whether feed-forward layer uses gated activations (wi_0 and wi_1).
        tie_word_embeddings: Whether input and output word embeddings are tied.
        pad_token_id: Padding token id.
        eos_token_id: End of sequence token id.
        decoder_start_token_id: First token for decoder input generation.
        classifier_dropout: Dropout probability for classification heads.
    """

    vocab_size: int = 128
    d_model: int = 1024
    d_kv: int = 64
    d_ff: int = 16384
    num_layers: int = 24
    num_decoder_layers: Optional[int] = None
    num_heads: int = 32
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    dense_act_fn: str = "relu"
    is_gated_act: bool = False
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    classifier_dropout: float = 0.0

    def __post_init__(self):
        if self.num_decoder_layers is None:
            self.num_decoder_layers = self.num_layers
        if "gated" in self.feed_forward_proj:
            self.is_gated_act = True

    # Property aliases for compatibility with different naming conventions
    @property
    def embed_dim(self) -> int:
        return self.d_model

    @property
    def key_value_dim(self) -> int:
        return self.d_kv

    @property
    def mlp_dim(self) -> int:
        return self.d_ff

    @property
    def layer_norm_eps(self) -> float:
        return self.layer_norm_epsilon

    @property
    def dropout(self) -> float:
        return self.dropout_rate

    @property
    def hidden_act(self) -> str:
        return self.dense_act_fn

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["embed_dim"] = self.embed_dim
        d["key_value_dim"] = self.key_value_dim
        d["mlp_dim"] = self.mlp_dim
        d["layer_norm_eps"] = self.layer_norm_eps
        d["dropout"] = self.dropout
        d["hidden_act"] = self.hidden_act
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProtT5Config":
        valid = {f.name for f in fields(cls)}
        mapped = dict(d)
        # Handle aliases
        if "embed_dim" in mapped and "d_model" not in mapped:
            mapped["d_model"] = mapped.pop("embed_dim")
        if "key_value_dim" in mapped and "d_kv" not in mapped:
            mapped["d_kv"] = mapped.pop("key_value_dim")
        if "mlp_dim" in mapped and "d_ff" not in mapped:
            mapped["d_ff"] = mapped.pop("mlp_dim")
        if "layer_norm_eps" in mapped and "layer_norm_epsilon" not in mapped:
            mapped["layer_norm_epsilon"] = mapped.pop("layer_norm_eps")
        if "dropout" in mapped and "dropout_rate" not in mapped:
            mapped["dropout_rate"] = mapped.pop("dropout")
        if "hidden_act" in mapped and "dense_act_fn" not in mapped:
            mapped["dense_act_fn"] = mapped.pop("hidden_act")

        filtered = {k: v for k, v in mapped.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_hf(cls, hf_config: Dict[str, Any]) -> "ProtT5Config":
        """Maps a HuggingFace T5/ProtT5 config dictionary to ProtT5Config."""
        feed_forward_proj = str(hf_config.get("feed_forward_proj", "relu"))
        is_gated = bool(
            hf_config.get("is_gated_act", False)
            or feed_forward_proj.startswith("gated")
        )
        dense_act = hf_config.get("dense_act_fn")
        if not dense_act:
            if is_gated and "gated" in feed_forward_proj:
                parts = feed_forward_proj.split("-")
                dense_act = parts[1] if len(parts) > 1 and parts[1] else "relu"
            else:
                dense_act = feed_forward_proj or "relu"

        num_layers = hf_config.get("num_layers", 24)
        num_decoder_layers = hf_config.get("num_decoder_layers", num_layers)

        return cls(
            vocab_size=hf_config.get("vocab_size", 128),
            d_model=hf_config.get("d_model", 1024),
            d_kv=hf_config.get("d_kv", 64),
            d_ff=hf_config.get("d_ff", 16384),
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=hf_config.get("num_heads", 32),
            relative_attention_num_buckets=hf_config.get(
                "relative_attention_num_buckets", 32
            ),
            relative_attention_max_distance=hf_config.get(
                "relative_attention_max_distance", 128
            ),
            dropout_rate=hf_config.get("dropout_rate", 0.1),
            layer_norm_epsilon=hf_config.get("layer_norm_epsilon", 1e-6),
            initializer_factor=hf_config.get("initializer_factor", 1.0),
            feed_forward_proj=feed_forward_proj,
            dense_act_fn=dense_act,
            is_gated_act=is_gated,
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
            pad_token_id=hf_config.get("pad_token_id", 0),
            eos_token_id=hf_config.get("eos_token_id", 1),
            decoder_start_token_id=hf_config.get("decoder_start_token_id", 0),
            classifier_dropout=hf_config.get("classifier_dropout", 0.0),
        )


# Backward-compatible alias
T5Config = ProtT5Config

