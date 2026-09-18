"""Configuration for ESM-2 protein language models in Keras 3.

Supports all standard ESM-2 checkpoints (8M, 35M, 150M, 650M, 3B, 15B)
as well as custom Transformer encoder architectures with Rotary Position
Embeddings (RoPE).
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


ESM2_PRESETS: Dict[str, Dict[str, Any]] = {
    "esm2_t6_8M_UR50D": {
        "num_hidden_layers": 6,
        "hidden_size": 320,
        "num_attention_heads": 20,
        "intermediate_size": 1280,
    },
    "esm2_t12_35M_UR50D": {
        "num_hidden_layers": 12,
        "hidden_size": 480,
        "num_attention_heads": 20,
        "intermediate_size": 1920,
    },
    "esm2_t30_150M_UR50D": {
        "num_hidden_layers": 30,
        "hidden_size": 640,
        "num_attention_heads": 20,
        "intermediate_size": 2560,
    },
    "esm2_t33_650M_UR50D": {
        "num_hidden_layers": 33,
        "hidden_size": 1280,
        "num_attention_heads": 20,
        "intermediate_size": 5120,
    },
    "esm2_t36_3B_UR50D": {
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "num_attention_heads": 40,
        "intermediate_size": 10240,
    },
    "esm2_t48_15B_UR50D": {
        "num_hidden_layers": 48,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "intermediate_size": 20480,
    },
}


@dataclass
class ESM2Config:
    """Architecture configuration for ESM-2 models.

    Args:
        vocab_size: Size of the vocabulary (default 33 for standard ESM-2).
        hidden_size: Dimensionality of the encoder layers and embeddings.
        num_hidden_layers: Number of Transformer encoder layers.
        num_attention_heads: Number of attention heads in each attention layer.
        intermediate_size: Dimensionality of the intermediate feed-forward layer.
        max_position_embeddings: Maximum sequence length supported by position embeddings.
        position_embedding_type: Position embedding type ("rotary" or "absolute").
        rope_theta: Base value for Rotary Position Embeddings frequency calculation.
        hidden_act: Activation function in intermediate layers ("gelu").
        layer_norm_eps: Epsilon value used for layer normalization.
        hidden_dropout_prob: Dropout probability for hidden representations.
        attention_probs_dropout_prob: Dropout probability for attention weights.
        token_dropout: Whether masked tokens are treated as dropout during training.
        mask_token_id: Vocabulary ID for <mask> token.
        pad_token_id: Vocabulary ID for <pad> token.
        cls_token_id: Vocabulary ID for <cls> token.
        eos_token_id: Vocabulary ID for <eos> token.
        unk_token_id: Vocabulary ID for <unk> token.
        tie_word_embeddings: Whether to tie input word embeddings and LM head projection.
        initializer_range: Standard deviation of truncated normal weight initializer.
        num_labels: Number of labels for sequence or token classification heads.
        classifier_dropout: Dropout probability for classification heads.
        emb_layer_norm_before: Whether to apply layer norm right after word embeddings (ESM-1b).
    """

    vocab_size: int = 33
    hidden_size: int = 320
    num_hidden_layers: int = 6
    num_attention_heads: int = 20
    intermediate_size: int = 1280
    max_position_embeddings: int = 1026
    position_embedding_type: str = "rotary"
    rope_theta: float = 10000.0
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    token_dropout: bool = True
    mask_token_id: int = 32
    pad_token_id: int = 1
    cls_token_id: int = 0
    eos_token_id: int = 2
    unk_token_id: int = 3
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    num_labels: int = 2
    classifier_dropout: Optional[float] = None
    emb_layer_norm_before: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    # Property aliases for flexibility
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
    def from_dict(cls, d: Dict[str, Any]) -> "ESM2Config":
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
    def from_preset(cls, preset_name: str, **kwargs: Any) -> "ESM2Config":
        """Instantiates an ESM2Config from a known checkpoint preset name."""
        clean_name = preset_name.replace("facebook/", "")
        if clean_name not in ESM2_PRESETS:
            valid_names = ", ".join(ESM2_PRESETS.keys())
            raise ValueError(
                f"Unknown ESM-2 preset '{preset_name}'. Valid presets are: {valid_names}"
            )
        preset_kwargs = dict(ESM2_PRESETS[clean_name])
        preset_kwargs.update(kwargs)
        return cls(**preset_kwargs)

    @classmethod
    def from_hf(cls, hf_config: Dict[str, Any]) -> "ESM2Config":
        """Maps a Hugging Face ESM config dictionary to ESM2Config."""
        return cls(
            vocab_size=hf_config.get("vocab_size", 33),
            hidden_size=hf_config.get("hidden_size", 320),
            num_hidden_layers=hf_config.get("num_hidden_layers", 6),
            num_attention_heads=hf_config.get("num_attention_heads", 20),
            intermediate_size=hf_config.get("intermediate_size", 1280),
            max_position_embeddings=hf_config.get("max_position_embeddings", 1026),
            position_embedding_type=hf_config.get("position_embedding_type", "rotary"),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            hidden_act=hf_config.get("hidden_act", "gelu"),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-5),
            hidden_dropout_prob=hf_config.get("hidden_dropout_prob", 0.0),
            attention_probs_dropout_prob=hf_config.get("attention_probs_dropout_prob", 0.0),
            token_dropout=hf_config.get("token_dropout", True),
            mask_token_id=hf_config.get("mask_token_id", 32),
            pad_token_id=hf_config.get("pad_token_id", 1),
            cls_token_id=hf_config.get("cls_token_id", 0),
            eos_token_id=hf_config.get("eos_token_id", 2),
            unk_token_id=hf_config.get("unk_token_id", 3),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
            initializer_range=hf_config.get("initializer_range", 0.02),
            num_labels=hf_config.get("num_labels", 2),
            classifier_dropout=hf_config.get("classifier_dropout", None),
            emb_layer_norm_before=hf_config.get("emb_layer_norm_before", False),
        )

