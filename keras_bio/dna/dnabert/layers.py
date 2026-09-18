"""Keras 3 layers for DNABERT models.

Implements standard Post-LN BERT architecture layers:
- DNABERTEmbeddings: Token, position, and segment (token type) embeddings with Post-LN
- DNABERTSelfAttention: Multi-head scaled dot-product self-attention
- DNABERTAttention: Self-attention sublayer with linear projection, residual, and Post-LN
- DNABERTIntermediate: Intermediate dense projection with GELU activation
- DNABERTOutput: Feed-forward output projection with residual and Post-LN
- DNABERTLayer: Full BERT Transformer encoder layer
- DNABERTPooler: [CLS] token extraction with tanh dense projection
- DNABERTLMHead: Masked language modeling prediction head with tied word embeddings
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras
from keras import layers, ops

from keras_bio.dna.dnabert.config import DNABERTConfig


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTEmbeddings(layers.Layer):
    """Combines token, position, and token-type embeddings with Post-LayerNorm."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.word_embeddings = layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="word_embeddings",
        )
        self.position_embeddings = layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="position_embeddings",
        )
        self.token_type_embeddings = layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="token_type_embeddings",
        )
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape: Any) -> None:
        if isinstance(input_shape, dict):
            ids_shape = input_shape.get("input_ids", (None, None))
        elif isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
            ids_shape = input_shape[0]
        else:
            ids_shape = input_shape

        self.word_embeddings.build(ids_shape)
        self.position_embeddings.build(ids_shape)
        self.token_type_embeddings.build(ids_shape)
        embed_shape = (*ids_shape, self.config.hidden_size)
        self.LayerNorm.build(embed_shape)
        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        shape = ops.shape(input_ids)
        batch_size = shape[0]
        seq_len = shape[1]

        if position_ids is None:
            position_ids = ops.broadcast_to(
                ops.expand_dims(ops.arange(0, seq_len, dtype="int32"), 0),
                (batch_size, seq_len),
            )

        if token_type_ids is None:
            token_type_ids = ops.zeros((batch_size, seq_len), dtype="int32")

        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = words + positions + token_types
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTEmbeddings":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTSelfAttention(layers.Layer):
    """Multi-head scaled dot-product self-attention for DNABERT."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.query = layers.Dense(self.hidden_size, use_bias=True, name="query")
        self.key = layers.Dense(self.hidden_size, use_bias=True, name="key")
        self.value = layers.Dense(self.hidden_size, use_bias=True, name="value")
        self.dropout = layers.Dropout(config.attention_probs_dropout_prob)

    def build(self, input_shape: Any) -> None:
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        seq_len = shape[1]

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = ops.reshape(q, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_attention_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product
        scale = 1.0 / (float(self.head_dim) ** 0.5)
        attn_scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

        if attention_mask is not None:
            mask_ndim = len(attention_mask.shape)
            if mask_ndim == 2:
                mask = ops.cast(attention_mask, attn_scores.dtype)
                mask_bias = (1.0 - ops.expand_dims(ops.expand_dims(mask, 1), 1)) * -1e9
                attn_scores = attn_scores + mask_bias
            else:
                attn_scores = attn_scores + ops.cast(attention_mask, attn_scores.dtype)

        attn_weights = ops.softmax(attn_scores, axis=-1)
        attn_weights_dropped = self.dropout(attn_weights, training=training)

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
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTSelfAttention":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTAttention(layers.Layer):
    """Self-attention sublayer with output dense projection, residual, and Post-LayerNorm."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.self_attn = DNABERTSelfAttention(config, name="self")
        self.dense = layers.Dense(config.hidden_size, use_bias=True, name="dense")
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape: Any) -> None:
        self.self_attn.build(input_shape)
        self.dense.build(input_shape)
        self.LayerNorm.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        if output_attentions:
            attn_output, attn_weights = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
                output_attentions=True,
            )
        else:
            attn_output = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
                output_attentions=False,
            )

        projected = self.dense(attn_output)
        projected = self.dropout(projected, training=training)
        # Post-LayerNorm: LayerNorm(residual + projected)
        output = self.LayerNorm(hidden_states + projected)

        if output_attentions:
            return output, attn_weights
        return output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTAttention":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTIntermediate(layers.Layer):
    """Feed-forward intermediate dense layer with GELU activation."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.dense = layers.Dense(
            config.intermediate_size, use_bias=True, name="dense"
        )

    def build(self, input_shape: Any) -> None:
        self.dense.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states: keras.KerasTensor) -> keras.KerasTensor:
        return ops.gelu(self.dense(hidden_states), approximate=False)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTIntermediate":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTOutput(layers.Layer):
    """Feed-forward output projection with residual connection and Post-LayerNorm."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.dense = layers.Dense(config.hidden_size, use_bias=True, name="dense")
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape: Any) -> None:
        self.dense.build(input_shape)
        out_shape = (*input_shape[:-1], self.config.hidden_size)
        self.LayerNorm.build(out_shape)
        super().build(input_shape)

    def call(
        self,
        intermediate_states: keras.KerasTensor,
        input_tensor: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        projected = self.dense(intermediate_states)
        projected = self.dropout(projected, training=training)
        # Post-LayerNorm: LayerNorm(residual + projected)
        return self.LayerNorm(input_tensor + projected)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTOutput":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTLayer(layers.Layer):
    """Complete Post-LN BERT Transformer encoder block for DNABERT."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.attention = DNABERTAttention(config, name="attention")
        self.intermediate = DNABERTIntermediate(config, name="intermediate")
        self.output_layer = DNABERTOutput(config, name="output")

    def build(self, input_shape: Any) -> None:
        self.attention.build(input_shape)
        self.intermediate.build(input_shape)
        inter_shape = (*input_shape[:-1], self.config.intermediate_size)
        self.output_layer.build(inter_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        if output_attentions:
            attn_output, attn_weights = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
                output_attentions=True,
            )
        else:
            attn_output = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
                output_attentions=False,
            )

        inter_output = self.intermediate(attn_output)
        layer_output = self.output_layer(
            inter_output, attn_output, training=training
        )

        if output_attentions:
            return layer_output, attn_weights
        return layer_output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTLayer":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTPooler(layers.Layer):
    """Extracts the [CLS] representation (index 0) and passes through a dense layer with tanh."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.dense = layers.Dense(
            config.hidden_size,
            activation="tanh",
            use_bias=True,
            name="dense",
        )

    def build(self, input_shape: Any) -> None:
        cls_shape = (input_shape[0], input_shape[-1])
        self.dense.build(cls_shape)
        super().build(input_shape)

    def call(self, hidden_states: keras.KerasTensor) -> keras.KerasTensor:
        # First token is [CLS]
        first_token = hidden_states[:, 0]
        return self.dense(first_token)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTPooler":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTPredictionHeadTransform(layers.Layer):
    """Dense + GELU + LayerNorm transformation for Masked LM predictions."""

    def __init__(
        self,
        config: DNABERTConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.dense = layers.Dense(config.hidden_size, use_bias=True, name="dense")
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )

    def build(self, input_shape: Any) -> None:
        self.dense.build(input_shape)
        self.LayerNorm.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states: keras.KerasTensor) -> keras.KerasTensor:
        x = self.dense(hidden_states)
        x = ops.gelu(x, approximate=False)
        return self.LayerNorm(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTPredictionHeadTransform":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTLMHead(layers.Layer):
    """Masked Language Modeling head for DNABERT with tied word embeddings."""

    def __init__(
        self,
        config: DNABERTConfig,
        tied_embedding_weights: Optional[keras.KerasTensor] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.transform = DNABERTPredictionHeadTransform(config, name="transform")
        self.tie_word_embeddings = config.tie_word_embeddings
        self._tied_weights = tied_embedding_weights

    def build(self, input_shape: Any) -> None:
        self.transform.build(input_shape)
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
        hidden_states: keras.KerasTensor,
        tied_embedding_weights: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        x = self.transform(hidden_states)

        if tied_embedding_weights is not None:
            embeddings_weights = tied_embedding_weights
        else:
            embeddings_weights = self._tied_weights

        if self.tie_word_embeddings and embeddings_weights is not None:
            logits = ops.matmul(x, ops.transpose(embeddings_weights)) + self.bias
        elif not self.tie_word_embeddings and hasattr(self, "decoder"):
            logits = self.decoder(x) + self.bias
        elif embeddings_weights is not None:
            logits = ops.matmul(x, ops.transpose(embeddings_weights)) + self.bias
        else:
            raise ValueError(
                "DNABERTLMHead requires tied_embedding_weights when tie_word_embeddings=True."
            )
        return logits

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTLMHead":
        cfg = DNABERTConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)

