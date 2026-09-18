"""Modular, serializable Keras 3 layers for ProtT5.

Supports all Keras backends (TensorFlow, PyTorch, JAX) using `keras.ops`.
Provides both standard ungated (ProtT5-XL) and gated (ProtT5-base-mt, T5-v1.1)
feed-forward projections.
"""

import math
from typing import Optional, Tuple

import keras
from keras import layers, ops

MASK_NEG = -1e9


def relative_position_bucket(
    relative_position,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
):
    """Bucket relative positions for T5 relative attention bias (memory - query)."""
    relative_buckets = 0
    if bidirectional:
        num_buckets = num_buckets // 2
        relative_buckets = ops.cast(relative_position > 0, "int32") * num_buckets
        relative_position = ops.abs(relative_position)
    else:
        relative_position = -ops.minimum(
            relative_position, ops.zeros_like(relative_position)
        )
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    safe = ops.cast(ops.maximum(relative_position, 1), "float32")
    relative_if_large = max_exact + ops.cast(
        ops.log(safe / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact),
        "int32",
    )
    relative_if_large = ops.minimum(relative_if_large, num_buckets - 1)
    relative_buckets = relative_buckets + ops.where(
        is_small, ops.cast(relative_position, "int32"), relative_if_large
    )
    return relative_buckets


def compute_relative_bias(
    table_or_embedding,
    query_length,
    key_length,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
):
    """Compute learned relative position bias `(1, num_heads, query_length, key_length)`."""
    context = ops.arange(query_length)[:, None]
    memory = ops.arange(key_length)[None, :]
    bucket = relative_position_bucket(
        memory - context, bidirectional, num_buckets, max_distance
    )
    if hasattr(table_or_embedding, "embeddings"):
        table = table_or_embedding.embeddings
        values = ops.take(table, bucket, axis=0)
    elif callable(table_or_embedding) and not isinstance(
        table_or_embedding, (ops.Tensor, keras.KerasTensor)
    ):
        values = table_or_embedding(bucket)
    else:
        values = ops.take(table_or_embedding, bucket, axis=0)
    return ops.transpose(values, (2, 0, 1))[None]  # (1, heads, query, key)


def get_t5_activation(act_fn: str):
    """Return activation callable corresponding to HuggingFace T5 activation string."""
    act_fn = act_fn.lower()
    if act_fn in ("gelu_new", "gelu_fast", "gated-gelu"):
        return lambda x: ops.gelu(x, approximate=True)
    elif act_fn in ("gelu", "gelu_python"):
        return lambda x: ops.gelu(x, approximate=False)
    elif act_fn in ("relu", "gated-relu"):
        return ops.relu
    elif act_fn in ("silu", "swish", "gated-silu"):
        return ops.silu
    else:
        return keras.activations.get(act_fn)


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5LayerNorm(layers.Layer):
    """T5-style RMSLayerNorm: scale only, no mean subtraction, no bias offset.

    y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.weight = self.add_weight(
            shape=(dim,),
            initializer="ones",
            name="weight",
        )
        super().build(input_shape)

    def call(self, hidden_states):
        variance = ops.mean(
            ops.square(ops.cast(hidden_states, "float32")),
            axis=-1,
            keepdims=True,
        )
        normed = hidden_states * ops.cast(
            ops.rsqrt(variance + self.eps), hidden_states.dtype
        )
        return self.weight * normed

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5PositionBias(layers.Layer):
    """Learned relative-position bias computation layer.

    Optionally applies causal masking (decoder self-attention) and key padding mask.
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
        causal: bool = False,
        rel_bias_embedding: Optional[layers.Embedding] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if rel_bias_embedding is not None:
            object.__setattr__(self, "rel_bias_embedding", rel_bias_embedding)
        else:
            self.rel_bias_embedding = None
        self.num_heads = int(num_heads)
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.bidirectional = bool(bidirectional)
        self.causal = bool(causal)

    def call(self, hidden, rel_bias_weights=None, attention_mask=None):
        seq = ops.shape(hidden)[1]
        bias_source = (
            rel_bias_weights
            if rel_bias_weights is not None
            else getattr(self, "rel_bias_embedding", None)
        )
        if bias_source is None:
            raise ValueError(
                "Either rel_bias_weights must be provided to call() or rel_bias_embedding "
                "must be provided in __init__."
            )
        bias = compute_relative_bias(
            bias_source,
            seq,
            seq,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        if self.causal:
            qi = ops.arange(seq)[:, None]
            ki = ops.arange(seq)[None, :]
            causal_mask = ops.where(ki <= qi, 0.0, MASK_NEG)
            bias = bias + ops.cast(causal_mask, "float32")[None, None]

        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            if len(ops.shape(am)) == 2:
                bias = bias + (1.0 - am)[:, None, None, :] * MASK_NEG
            elif len(ops.shape(am)) == 3:
                bias = bias + (1.0 - am)[:, None, :, :] * MASK_NEG
            elif len(ops.shape(am)) == 4:
                bias = bias + (1.0 - am) * MASK_NEG
        return bias

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            if isinstance(input_shape[0], tuple):
                input_shape = input_shape[0]
        batch = input_shape[0]
        seq = input_shape[1]
        return (batch, self.num_heads, seq, seq)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "num_buckets": self.num_buckets,
                "max_distance": self.max_distance,
                "bidirectional": self.bidirectional,
                "causal": self.causal,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5TiedHead(layers.Layer):
    """Weightless LM projection head projecting with the shared embedding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs can be (hidden, embedding_weights)
        if isinstance(inputs, (list, tuple)):
            hidden, embedding_weights = inputs
        else:
            raise ValueError(
                "ProtT5TiedHead expects inputs as [hidden_states, embedding_weights]"
            )
        kernel = ops.transpose(ops.cast(embedding_weights, hidden.dtype))
        return ops.matmul(hidden, kernel)

    def compute_output_shape(self, input_shape):
        hidden_shape, emb_shape = input_shape
        return tuple(hidden_shape[:-1]) + (emb_shape[0],)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5Attention(layers.Layer):
    """Multi-head attention layer for ProtT5.

    Uses bias-free linear projections (q, k, v, o) and supports relative
    position bias added to raw attention scores.
    """

    def __init__(
        self,
        embed_dim: int,
        key_value_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        prefix: str = "attn",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.key_value_dim = int(key_value_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.prefix = prefix
        self.inner_dim = self.num_heads * self.key_value_dim

        self.q = layers.Dense(self.inner_dim, use_bias=False, name=f"{prefix}_q")
        self.k = layers.Dense(self.inner_dim, use_bias=False, name=f"{prefix}_k")
        self.v = layers.Dense(self.inner_dim, use_bias=False, name=f"{prefix}_v")
        self.o = layers.Dense(self.embed_dim, use_bias=False, name=f"{prefix}_o")
        self.dropout_layer = layers.Dropout(self.dropout, name=f"{prefix}_dropout")

    def build(self, input_shape):
        self.q.build(input_shape)
        self.k.build(input_shape)
        self.v.build(input_shape)
        inter_shape = tuple(input_shape[:-1]) + (self.inner_dim,)
        self.o.build(inter_shape)
        super().build(input_shape)

    def split_heads(self, x):
        batch = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        x = ops.reshape(x, (batch, seq_len, self.num_heads, self.key_value_dim))
        return ops.transpose(x, (0, 2, 1, 3))  # (batch, heads, seq, d_kv)

    def call(
        self,
        hidden_states,
        position_bias=None,
        key_value_states=None,
        training: Optional[bool] = None,
    ):
        batch = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]
        current_kv = key_value_states if key_value_states is not None else hidden_states

        query = self.split_heads(self.q(hidden_states))
        key = self.split_heads(self.k(current_kv))
        value = self.split_heads(self.v(current_kv))

        # Raw scores: (batch, heads, q_len, k_len)
        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        if position_bias is not None:
            scores = scores + ops.cast(position_bias, scores.dtype)

        weights = ops.softmax(ops.cast(scores, "float32"), axis=-1)
        weights = ops.cast(weights, hidden_states.dtype)
        weights = self.dropout_layer(weights, training=training)

        # Context: (batch, heads, q_len, d_kv)
        out = ops.matmul(weights, value)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch, seq_len, self.inner_dim))
        return self.o(out)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "key_value_dim": self.key_value_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5DenseActDense(layers.Layer):
    """Ungated feed-forward projection: wi -> act -> dropout -> wo.

    Used by ProtT5-XL and original T5.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        hidden_act: str = "relu",
        dropout: float = 0.1,
        prefix: str = "ff",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.mlp_dim = int(mlp_dim)
        self.hidden_act = hidden_act
        self.dropout = float(dropout)
        self.prefix = prefix

        self.wi = layers.Dense(self.mlp_dim, use_bias=False, name=f"{prefix}_wi")
        self.act = get_t5_activation(self.hidden_act)
        self.dropout_layer = layers.Dropout(self.dropout, name=f"{prefix}_dropout")
        self.wo = layers.Dense(self.embed_dim, use_bias=False, name=f"{prefix}_wo")

    def build(self, input_shape):
        self.wi.build(input_shape)
        inter_shape = tuple(input_shape[:-1]) + (self.mlp_dim,)
        self.wo.build(inter_shape)
        super().build(input_shape)

    def call(self, hidden_states, training: Optional[bool] = None):
        h = self.wi(hidden_states)
        h = self.act(h)
        h = self.dropout_layer(h, training=training)
        return self.wo(h)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "hidden_act": self.hidden_act,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5DenseGatedDense(layers.Layer):
    """Gated feed-forward projection: (act(wi_0) * wi_1) -> dropout -> wo.

    Used by ProtT5-base-mt, T5-v1.1, and mT5.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        hidden_act: str = "gelu_new",
        dropout: float = 0.1,
        prefix: str = "ff",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.mlp_dim = int(mlp_dim)
        self.hidden_act = hidden_act
        self.dropout = float(dropout)
        self.prefix = prefix

        self.wi_0 = layers.Dense(self.mlp_dim, use_bias=False, name=f"{prefix}_wi_0")
        self.wi_1 = layers.Dense(self.mlp_dim, use_bias=False, name=f"{prefix}_wi_1")
        self.act = get_t5_activation(self.hidden_act)
        self.dropout_layer = layers.Dropout(self.dropout, name=f"{prefix}_dropout")
        self.wo = layers.Dense(self.embed_dim, use_bias=False, name=f"{prefix}_wo")

    def build(self, input_shape):
        self.wi_0.build(input_shape)
        self.wi_1.build(input_shape)
        inter_shape = tuple(input_shape[:-1]) + (self.mlp_dim,)
        self.wo.build(inter_shape)
        super().build(input_shape)

    def call(self, hidden_states, training: Optional[bool] = None):
        h_gate = self.act(self.wi_0(hidden_states))
        h_linear = self.wi_1(hidden_states)
        h = h_gate * h_linear
        h = self.dropout_layer(h, training=training)
        return self.wo(h)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "hidden_act": self.hidden_act,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5FeedForwardLayer(layers.Layer):
    """FeedForward wrapper layer with pre-LayerNorm and residual connection."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        hidden_act: str = "relu",
        is_gated_act: bool = False,
        eps: float = 1e-6,
        dropout: float = 0.1,
        prefix: str = "ff",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.mlp_dim = int(mlp_dim)
        self.hidden_act = hidden_act
        self.is_gated_act = bool(is_gated_act)
        self.eps = float(eps)
        self.dropout = float(dropout)
        self.prefix = prefix

        self.layer_norm = ProtT5LayerNorm(eps, name=f"{prefix}_ln")
        if self.is_gated_act:
            self.ff = ProtT5DenseGatedDense(
                embed_dim=self.embed_dim,
                mlp_dim=self.mlp_dim,
                hidden_act=self.hidden_act,
                dropout=self.dropout,
                prefix=prefix,
                name=f"{prefix}_dense",
            )
        else:
            self.ff = ProtT5DenseActDense(
                embed_dim=self.embed_dim,
                mlp_dim=self.mlp_dim,
                hidden_act=self.hidden_act,
                dropout=self.dropout,
                prefix=prefix,
                name=f"{prefix}_dense",
            )
        self.dropout_layer = layers.Dropout(self.dropout, name=f"{prefix}_res_dropout")

    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.ff.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states, training: Optional[bool] = None):
        normed = self.layer_norm(hidden_states)
        forwarded = self.ff(normed, training=training)
        forwarded = self.dropout_layer(forwarded, training=training)
        return hidden_states + forwarded

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "hidden_act": self.hidden_act,
                "is_gated_act": self.is_gated_act,
                "eps": self.eps,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5SelfAttentionLayer(layers.Layer):
    """Self-attention sublayer with pre-LayerNorm and residual connection."""

    def __init__(
        self,
        embed_dim: int,
        key_value_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        dropout: float = 0.1,
        prefix: str = "self",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.key_value_dim = int(key_value_dim)
        self.num_heads = int(num_heads)
        self.eps = float(eps)
        self.dropout = float(dropout)
        self.prefix = prefix

        self.layer_norm = ProtT5LayerNorm(eps, name=f"{prefix}_ln")
        self.attention = ProtT5Attention(
            embed_dim=self.embed_dim,
            key_value_dim=self.key_value_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            prefix=prefix,
            name=f"{prefix}_attn",
        )
        self.dropout_layer = layers.Dropout(self.dropout, name=f"{prefix}_res_dropout")

    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.attention.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states, position_bias=None, training: Optional[bool] = None):
        normed = self.layer_norm(hidden_states)
        attn = self.attention(normed, position_bias=position_bias, training=training)
        attn = self.dropout_layer(attn, training=training)
        return hidden_states + attn

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "key_value_dim": self.key_value_dim,
                "num_heads": self.num_heads,
                "eps": self.eps,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5CrossAttentionLayer(layers.Layer):
    """Cross-attention sublayer with pre-LayerNorm and residual connection."""

    def __init__(
        self,
        embed_dim: int,
        key_value_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        dropout: float = 0.1,
        prefix: str = "cross",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.key_value_dim = int(key_value_dim)
        self.num_heads = int(num_heads)
        self.eps = float(eps)
        self.dropout = float(dropout)
        self.prefix = prefix

        self.layer_norm = ProtT5LayerNorm(eps, name=f"{prefix}_ln")
        self.attention = ProtT5Attention(
            embed_dim=self.embed_dim,
            key_value_dim=self.key_value_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            prefix=prefix,
            name=f"{prefix}_attn",
        )
        self.dropout_layer = layers.Dropout(self.dropout, name=f"{prefix}_res_dropout")

    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.attention.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        encoder_hidden_states,
        position_bias=None,
        training: Optional[bool] = None,
    ):
        normed = self.layer_norm(hidden_states)
        attn = self.attention(
            normed,
            position_bias=position_bias,
            key_value_states=encoder_hidden_states,
            training=training,
        )
        attn = self.dropout_layer(attn, training=training)
        return hidden_states + attn

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "key_value_dim": self.key_value_dim,
                "num_heads": self.num_heads,
                "eps": self.eps,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5EncoderBlock(layers.Layer):
    """Full ProtT5 Transformer Encoder Block (Self-Attention + FeedForward)."""

    def __init__(
        self,
        embed_dim: int,
        key_value_dim: int,
        num_heads: int,
        mlp_dim: int,
        hidden_act: str = "relu",
        is_gated_act: bool = False,
        eps: float = 1e-6,
        dropout: float = 0.1,
        prefix: str = "enc_0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.key_value_dim = int(key_value_dim)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.hidden_act = hidden_act
        self.is_gated_act = bool(is_gated_act)
        self.eps = float(eps)
        self.dropout = float(dropout)
        self.prefix = prefix

        self.self_attention = ProtT5SelfAttentionLayer(
            embed_dim=self.embed_dim,
            key_value_dim=self.key_value_dim,
            num_heads=self.num_heads,
            eps=self.eps,
            dropout=self.dropout,
            prefix=f"{prefix}_attn",
            name=f"{prefix}_self_attn",
        )
        self.ff = ProtT5FeedForwardLayer(
            embed_dim=self.embed_dim,
            mlp_dim=self.mlp_dim,
            hidden_act=self.hidden_act,
            is_gated_act=self.is_gated_act,
            eps=self.eps,
            dropout=self.dropout,
            prefix=f"{prefix}_ff",
            name=f"{prefix}_feed_forward",
        )

    def build(self, input_shape):
        self.self_attention.build(input_shape)
        self.ff.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states, position_bias=None, training: Optional[bool] = None):
        h = self.self_attention(hidden_states, position_bias=position_bias, training=training)
        return self.ff(h, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "key_value_dim": self.key_value_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "hidden_act": self.hidden_act,
                "is_gated_act": self.is_gated_act,
                "eps": self.eps,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5DecoderBlock(layers.Layer):
    """Full ProtT5 Transformer Decoder Block (Self-Attention + Cross-Attention + FeedForward)."""

    def __init__(
        self,
        embed_dim: int,
        key_value_dim: int,
        num_heads: int,
        mlp_dim: int,
        hidden_act: str = "relu",
        is_gated_act: bool = False,
        eps: float = 1e-6,
        dropout: float = 0.1,
        prefix: str = "dec_0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.key_value_dim = int(key_value_dim)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.hidden_act = hidden_act
        self.is_gated_act = bool(is_gated_act)
        self.eps = float(eps)
        self.dropout = float(dropout)
        self.prefix = prefix

        self.self_attention = ProtT5SelfAttentionLayer(
            embed_dim=self.embed_dim,
            key_value_dim=self.key_value_dim,
            num_heads=self.num_heads,
            eps=self.eps,
            dropout=self.dropout,
            prefix=f"{prefix}_self",
            name=f"{prefix}_self_attn",
        )
        self.cross_attention = ProtT5CrossAttentionLayer(
            embed_dim=self.embed_dim,
            key_value_dim=self.key_value_dim,
            num_heads=self.num_heads,
            eps=self.eps,
            dropout=self.dropout,
            prefix=f"{prefix}_cross",
            name=f"{prefix}_cross_attn",
        )
        self.ff = ProtT5FeedForwardLayer(
            embed_dim=self.embed_dim,
            mlp_dim=self.mlp_dim,
            hidden_act=self.hidden_act,
            is_gated_act=self.is_gated_act,
            eps=self.eps,
            dropout=self.dropout,
            prefix=f"{prefix}_ff",
            name=f"{prefix}_feed_forward",
        )

    def build(self, input_shape):
        self.self_attention.build(input_shape)
        self.cross_attention.build(input_shape)
        self.ff.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        self_position_bias=None,
        encoder_hidden_states=None,
        cross_position_bias=None,
        training: Optional[bool] = None,
    ):
        h = self.self_attention(
            hidden_states, position_bias=self_position_bias, training=training
        )
        if encoder_hidden_states is not None:
            h = self.cross_attention(
                h,
                encoder_hidden_states,
                position_bias=cross_position_bias,
                training=training,
            )
        return self.ff(h, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "key_value_dim": self.key_value_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "hidden_act": self.hidden_act,
                "is_gated_act": self.is_gated_act,
                "eps": self.eps,
                "dropout": self.dropout,
                "prefix": self.prefix,
            }
        )
        return config


# Backward-compatible aliases
T5LayerNorm = ProtT5LayerNorm
T5PositionBias = ProtT5PositionBias
T5Attention = ProtT5Attention
T5DenseActDense = ProtT5DenseActDense
T5DenseGatedDense = ProtT5DenseGatedDense
T5FeedForwardLayer = ProtT5FeedForwardLayer
T5SelfAttentionLayer = ProtT5SelfAttentionLayer
T5CrossAttentionLayer = ProtT5CrossAttentionLayer
T5EncoderBlock = ProtT5EncoderBlock
T5DecoderBlock = ProtT5DecoderBlock
T5TiedHead = ProtT5TiedHead

