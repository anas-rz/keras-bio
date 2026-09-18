"""Keras 3 layers for ESM-2 protein language models.

Implements Rotary Position Embeddings (RoPE), Pre-LN Transformer encoder
blocks, Contact Prediction Head with Average Product Correction (APC),
and Masked Language Model / Sequence Classification heads.
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras
from keras import layers, ops

from keras_bio.protein.esm2.config import ESM2Config


def rotate_half(x: keras.KerasTensor) -> keras.KerasTensor:
    """Rotates half the hidden dimensions of the input tensor."""
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: keras.KerasTensor,
    k: keras.KerasTensor,
    cos: keras.KerasTensor,
    sin: keras.KerasTensor,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim).
        cos: Cosine tensor of shape (batch, seq_len, head_dim).
        sin: Sine tensor of shape (batch, seq_len, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    cos_exp = ops.expand_dims(cos, 1)  # (batch, 1, seq_len, head_dim)
    sin_exp = ops.expand_dims(sin, 1)  # (batch, 1, seq_len, head_dim)

    q_rot = (q * cos_exp) + (rotate_half(q) * sin_exp)
    k_rot = (k * cos_exp) + (rotate_half(k) * sin_exp)
    return q_rot, k_rot


def symmetrize(x: keras.KerasTensor) -> keras.KerasTensor:
    """Makes a tensor symmetric across its last two dimensions."""
    return x + ops.transpose(x, (0, 1, 3, 2))


def average_product_correct(x: keras.KerasTensor) -> keras.KerasTensor:
    """Performs Average Product Correction (APC) for contact prediction."""
    a1 = ops.sum(x, axis=-1, keepdims=True)
    a2 = ops.sum(x, axis=-2, keepdims=True)
    a12 = ops.sum(x, axis=(-1, -2), keepdims=True)
    avg = (a1 * a2) / ops.maximum(a12, 1e-12)
    return x - avg


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2RotaryEmbedding(layers.Layer):
    """Computes Rotary Position Embeddings (RoPE) for ESM-2."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.base = float(base)

    def build(self, input_shape: Any) -> None:
        inv_freq = 1.0 / (
            self.base ** (ops.arange(0, self.dim, 2, dtype="float32") / self.dim)
        )
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(self.dim // 2,),
            initializer=keras.initializers.Constant(ops.convert_to_numpy(inv_freq)),
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        position_ids: Optional[keras.KerasTensor] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Computes RoPE cosine and sine tensors.

        Args:
            x: Input hidden states of shape (batch, seq_len, hidden_size).
            position_ids: Optional position indices of shape (batch, seq_len).

        Returns:
            Tuple of (cos, sin), each of shape (batch, seq_len, dim).
        """
        shape = ops.shape(x)
        batch_size = shape[0]
        seq_len = shape[1]

        if position_ids is None:
            position_ids = ops.broadcast_to(
                ops.expand_dims(ops.arange(0, seq_len, dtype="float32"), 0),
                (batch_size, seq_len),
            )
        else:
            position_ids = ops.cast(position_ids, "float32")

        # inv_freq: (dim // 2,)
        # pos: (batch, seq_len)
        # freqs: (batch, seq_len, dim // 2)
        freqs = ops.expand_dims(position_ids, -1) * ops.expand_dims(
            ops.expand_dims(self.inv_freq, 0), 0
        )
        emb = ops.concatenate([freqs, freqs], axis=-1)  # (batch, seq_len, dim)
        cos = ops.cast(ops.cos(emb), x.dtype)
        sin = ops.cast(ops.sin(emb), x.dtype)
        return cos, sin

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"dim": self.dim, "base": self.base})
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2SelfAttention(layers.Layer):
    """Multi-Head Self Attention with Rotary Position Embeddings for ESM-2."""

    def __init__(
        self,
        config: ESM2Config,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.query = layers.Dense(self.hidden_size, use_bias=True, name="query")
        self.key = layers.Dense(self.hidden_size, use_bias=True, name="key")
        self.value = layers.Dense(self.hidden_size, use_bias=True, name="value")
        self.dropout = layers.Dropout(self.attention_probs_dropout_prob)

    def build(self, input_shape: Any) -> None:
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_embeddings: Optional[
            Tuple[keras.KerasTensor, keras.KerasTensor]
        ] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        seq_len = shape[1]

        # Projections: (batch, seq_len, hidden_size) -> (batch, heads, seq_len, head_dim)
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = ops.reshape(q, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_attention_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # ESM scales query down by sqrt(head_dim) before applying RoPE
        q = q * (float(self.head_dim) ** -0.5)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores: (batch, heads, seq_len, seq_len)
        attn_scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        if attention_mask is not None:
            mask_ndim = len(attention_mask.shape)
            if mask_ndim == 2:
                mask = ops.cast(attention_mask, attn_scores.dtype)
                # Expand (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask_bias = (1.0 - ops.expand_dims(ops.expand_dims(mask, 1), 1)) * -1e9
                attn_scores = attn_scores + mask_bias
            else:
                attn_scores = attn_scores + ops.cast(attention_mask, attn_scores.dtype)

        attn_weights = ops.softmax(attn_scores, axis=-1)
        attn_weights_dropped = self.dropout(attn_weights, training=training)

        # Context: (batch, heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        context = ops.matmul(attn_weights_dropped, v)
        context = ops.transpose(context, (0, 2, 1, 3))
        context = ops.reshape(context, (batch_size, seq_len, self.hidden_size))

        if output_attentions:
            return context, attn_weights
        return context

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2SelfAttention":
        cfg = ESM2Config.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2Attention(layers.Layer):
    """Pre-LayerNorm Attention module with residual connection for ESM-2."""

    def __init__(
        self,
        config: ESM2Config,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.self_attn = ESM2SelfAttention(config, name="self")
        self.output_dense = layers.Dense(
            config.hidden_size, use_bias=True, name="output_dense"
        )
        self.output_dropout = layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape: Any) -> None:
        self.LayerNorm.build(input_shape)
        self.self_attn.build(input_shape)
        self.output_dense.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_embeddings: Optional[
            Tuple[keras.KerasTensor, keras.KerasTensor]
        ] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        # Pre-LayerNorm
        normed = self.LayerNorm(hidden_states)

        if output_attentions:
            attn_output, attn_weights = self.self_attn(
                normed,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                training=training,
                output_attentions=True,
            )
        else:
            attn_output = self.self_attn(
                normed,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                training=training,
                output_attentions=False,
            )

        attn_output = self.output_dense(attn_output)
        attn_output = self.output_dropout(attn_output, training=training)
        hidden_states = hidden_states + attn_output

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2Attention":
        cfg = ESM2Config.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2FeedForward(layers.Layer):
    """Pre-LayerNorm Feed-Forward module with residual connection for ESM-2."""

    def __init__(
        self,
        config: ESM2Config,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.intermediate_dense = layers.Dense(
            config.intermediate_size, use_bias=True, name="intermediate_dense"
        )
        self.output_dense = layers.Dense(
            config.hidden_size, use_bias=True, name="output_dense"
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape: Any) -> None:
        self.LayerNorm.build(input_shape)
        self.intermediate_dense.build(input_shape)
        inter_shape = list(input_shape)
        inter_shape[-1] = self.config.intermediate_size
        self.output_dense.build(tuple(inter_shape))
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        normed = self.LayerNorm(hidden_states)
        intermediate = self.intermediate_dense(normed)
        intermediate = ops.gelu(intermediate, approximate=False)
        output = self.output_dense(intermediate)
        output = self.dropout(output, training=training)
        return hidden_states + output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2FeedForward":
        cfg = ESM2Config.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2EncoderLayer(layers.Layer):
    """Complete ESM-2 Transformer encoder layer."""

    def __init__(
        self,
        config: ESM2Config,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.attention = ESM2Attention(config, name="attention")
        self.feed_forward = ESM2FeedForward(config, name="feed_forward")

    def build(self, input_shape: Any) -> None:
        self.attention.build(input_shape)
        self.feed_forward.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_embeddings: Optional[
            Tuple[keras.KerasTensor, keras.KerasTensor]
        ] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        if output_attentions:
            hidden_states, attn_weights = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                training=training,
                output_attentions=True,
            )
        else:
            hidden_states = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                training=training,
                output_attentions=False,
            )

        hidden_states = self.feed_forward(hidden_states, training=training)

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2EncoderLayer":
        cfg = ESM2Config.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2ContactPredictionHead(layers.Layer):
    """Predicts residue-residue contact maps from attention matrices across layers and heads."""

    def __init__(
        self,
        in_features: int,
        eos_idx: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.eos_idx = eos_idx
        self.regression = layers.Dense(1, use_bias=True, name="regression")

    def build(self, input_shape: Any) -> None:
        self.regression.build((None, None, None, self.in_features))
        super().build(input_shape)

    def call(
        self,
        tokens: keras.KerasTensor,
        attentions: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Predicts contact map for residues.

        Args:
            tokens: Token IDs of shape (batch, seq_len).
            attentions: Stacked attention maps of shape
                (batch, num_layers, num_heads, seq_len, seq_len).

        Returns:
            Contact map tensor of shape (batch, seq_len - 2, seq_len - 2).
        """
        shape = ops.shape(attentions)
        batch_size = shape[0]
        num_layers_cnt = shape[1]
        num_heads_cnt = shape[2]
        seq_len = shape[3]

        # Zero out attentions at EOS tokens
        eos_mask = ops.cast(ops.not_equal(tokens, self.eos_idx), attentions.dtype)
        eos_mask_2d = ops.expand_dims(eos_mask, 1) * ops.expand_dims(eos_mask, 2)
        attentions = attentions * ops.expand_dims(
            ops.expand_dims(eos_mask_2d, 1), 1
        )

        # Remove EOS (last position) and CLS (first position)
        attentions = attentions[:, :, :, 1:-1, 1:-1]
        new_seq_len = seq_len - 2

        # Flatten layers and heads into channels
        attentions = ops.reshape(
            attentions, (batch_size, num_layers_cnt * num_heads_cnt, new_seq_len, new_seq_len)
        )

        # Symmetrize and APC
        attentions = symmetrize(attentions)
        attentions = average_product_correct(attentions)

        # Channel-last: (batch, new_seq_len, new_seq_len, num_layers * num_heads)
        features = ops.transpose(attentions, (0, 2, 3, 1))

        # Regression + Sigmoid
        logits = self.regression(features)
        contacts = ops.squeeze(ops.sigmoid(logits), axis=-1)
        return contacts

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_features": self.in_features,
            "eos_idx": self.eos_idx,
        })
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2LMHead(layers.Layer):
    """Masked Language Modeling head for ESM-2."""

    def __init__(
        self,
        config: ESM2Config,
        tied_embedding_weights: Optional[keras.KerasTensor] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.dense = layers.Dense(config.hidden_size, use_bias=True, name="dense")
        self.layer_norm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layer_norm"
        )
        self.tie_word_embeddings = config.tie_word_embeddings
        self._tied_weights = tied_embedding_weights

    def build(self, input_shape: Any) -> None:
        self.dense.build(input_shape)
        self.layer_norm.build(input_shape)
        self.bias = self.add_weight(
            name="bias",
            shape=(self.config.vocab_size,),
            initializer="zeros",
            trainable=True,
        )
        if not self.tie_word_embeddings:
            self.decoder = layers.Dense(
                self.config.vocab_size, use_bias=False, name="decoder"
            )
            self.decoder.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        features: keras.KerasTensor,
        tied_embedding_weights: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        x = self.dense(features)
        x = ops.gelu(x, approximate=False)
        x = self.layer_norm(x)

        if tied_embedding_weights is not None:
            embeddings_weights = tied_embedding_weights
        else:
            embeddings_weights = self._tied_weights
        if self.tie_word_embeddings and embeddings_weights is not None:
            # Tied projection: x @ embeddings.T + bias
            logits = ops.matmul(x, ops.transpose(embeddings_weights)) + self.bias
        elif not self.tie_word_embeddings and hasattr(self, "decoder"):
            logits = self.decoder(x) + self.bias
        elif embeddings_weights is not None:
            logits = ops.matmul(x, ops.transpose(embeddings_weights)) + self.bias
        else:
            raise ValueError(
                "ESM2LMHead requires tied_embedding_weights when tie_word_embeddings=True."
            )
        return logits

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2LMHead":
        cfg = ESM2Config.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2ClassificationHead(layers.Layer):
    """Sequence classification head for ESM-2."""

    def __init__(
        self,
        config: ESM2Config,
        num_labels: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.num_labels = num_labels or config.num_labels
        dropout_prob = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = layers.Dropout(dropout_prob)
        self.dense = layers.Dense(config.hidden_size, use_bias=True, name="dense")
        self.out_proj = layers.Dense(self.num_labels, use_bias=True, name="out_proj")

    def build(self, input_shape: Any) -> None:
        cls_shape = (input_shape[0], input_shape[-1])
        self.dense.build(cls_shape)
        self.out_proj.build(cls_shape)
        super().build(input_shape)

    def call(
        self,
        features: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # Take <cls> token at index 0
        x = features[:, 0, :]
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = ops.tanh(x)
        x = self.dropout(x, training=training)
        return self.out_proj(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "config": self.config.to_dict(),
            "num_labels": self.num_labels,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2ClassificationHead":
        cfg = ESM2Config.from_dict(config.pop("config"))
        return cls(config=cfg, **config)
