"""ESM-2 models for keras-bio.

Modular Keras 3 models supporting all backends (TensorFlow, PyTorch, JAX):
- ESM2Model: Transformer encoder backbone with RoPE and contact prediction head
- ESM2ForMaskedLM: Masked language modeling for zero-shot mutation effect prediction
- ESM2ForSequenceClassification: Protein-level classification/regression head
- ESM2ForTokenClassification: Residue-level classification head
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import layers, ops

from keras_bio.protein.esm2.config import ESM2Config
from keras_bio.protein.esm2.layers import (
    ESM2ClassificationHead,
    ESM2ContactPredictionHead,
    ESM2EncoderLayer,
    ESM2LMHead,
    ESM2RotaryEmbedding,
)


def _unpack_inputs(inputs: Any) -> Tuple[Any, Any, Any]:
    """Extracts input_ids and attention_mask from inputs."""
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)
        position_ids = inputs.get("position_ids", None)
    elif isinstance(inputs, (list, tuple)):
        input_ids = inputs[0]
        attention_mask = inputs[1] if len(inputs) > 1 else None
        position_ids = inputs[2] if len(inputs) > 2 else None
    else:
        input_ids = inputs
        attention_mask = None
        position_ids = None
    return input_ids, attention_mask, position_ids


Tuple_inputs = Any


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2Model(keras.Model):
    """ESM-2 Transformer Encoder backbone with RoPE and contact prediction head.

    Args:
        config: Optional ESM2Config instance. If not provided, defaults to
            ESM-2 8M preset (`esm2_t6_8M_UR50D`).
    """

    def __init__(self, config: Optional[ESM2Config] = None, **kwargs: Any):
        if config is None:
            valid = {f.name for f in ESM2Config.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = ESM2Config(**cfg_kwargs) if cfg_kwargs else ESM2Config()

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

        if config.emb_layer_norm_before:
            self.emb_layer_norm_before = layers.LayerNormalization(
                epsilon=config.layer_norm_eps, name="emb_layer_norm_before"
            )
        else:
            self.emb_layer_norm_before = None

        self.rotary_embeddings = ESM2RotaryEmbedding(
            dim=config.head_dim,
            base=config.rope_theta,
            name="rotary_embeddings",
        )

        self.encoder_layers = [
            ESM2EncoderLayer(config, name=f"layer_{i}")
            for i in range(config.num_hidden_layers)
        ]

        self.emb_layer_norm_after = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="emb_layer_norm_after"
        )

        self.contact_head = ESM2ContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            eos_idx=config.eos_token_id,
            name="contact_head",
        )

    def build(self, input_shape: Any) -> None:
        if isinstance(input_shape, dict):
            ids_shape = input_shape["input_ids"]
        elif isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            ids_shape = input_shape[0]
        else:
            ids_shape = input_shape

        self.word_embeddings.build(ids_shape)
        embed_shape = (*ids_shape, self.config.hidden_size)

        if self.emb_layer_norm_before is not None:
            self.emb_layer_norm_before.build(embed_shape)

        self.rotary_embeddings.build(embed_shape)

        for layer in self.encoder_layers:
            layer.build(embed_shape)

        self.emb_layer_norm_after.build(embed_shape)
        self.contact_head.build(embed_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        input_ids, in_mask, in_pos = _unpack_inputs(inputs)
        if attention_mask is None:
            attention_mask = in_mask
        if position_ids is None:
            position_ids = in_pos

        hidden_states = self.word_embeddings(input_ids)

        if self.config.token_dropout:
            mask_token_id = self.config.mask_token_id
            is_masked = ops.cast(ops.equal(input_ids, mask_token_id), hidden_states.dtype)
            hidden_states = hidden_states * (1.0 - ops.expand_dims(is_masked, -1))

            mask_ratio_train = 0.15 * 0.8
            if attention_mask is not None:
                src_lengths = ops.cast(ops.sum(attention_mask, axis=-1), "float32")
            else:
                src_lengths = ops.cast(ops.shape(input_ids)[1], "float32")
            mask_ratio_observed = ops.cast(ops.sum(is_masked, axis=-1), "float32") / ops.maximum(src_lengths, 1.0)
            scale = (1.0 - mask_ratio_train) / ops.maximum(1.0 - mask_ratio_observed, 1e-8)
            hidden_states = hidden_states * ops.expand_dims(ops.expand_dims(ops.cast(scale, hidden_states.dtype), -1), -1)

        if self.emb_layer_norm_before is not None:
            hidden_states = self.emb_layer_norm_before(hidden_states)

        if attention_mask is not None:
            mask_2d = ops.cast(attention_mask, hidden_states.dtype)
            hidden_states = hidden_states * ops.expand_dims(mask_2d, -1)

        cos, sin = self.rotary_embeddings(hidden_states, position_ids=position_ids)

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if output_attentions:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=(cos, sin),
                    training=training,
                    output_attentions=True,
                )
                all_attentions.append(attn_weights)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=(cos, sin),
                    training=training,
                    output_attentions=False,
                )

        hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        result: Dict[str, Any] = {"last_hidden_state": hidden_states}
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        if output_attentions:
            result["attentions"] = all_attentions
        return result

    def predict_contacts(
        self,
        tokens: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Predicts residue-residue contact probability maps.

        Args:
            tokens: Input token IDs of shape (batch, seq_len).
            attention_mask: Optional attention mask of shape (batch, seq_len).

        Returns:
            Contact probabilities tensor of shape (batch, seq_len - 2, seq_len - 2).
        """
        if attention_mask is None:
            attention_mask = ops.cast(
                ops.not_equal(tokens, self.config.pad_token_id), "int32"
            )

        outputs = self(
            tokens,
            attention_mask=attention_mask,
            output_attentions=True,
            training=False,
        )
        attns = outputs["attentions"]
        # Stack: (batch, num_layers, num_heads, seq_len, seq_len)
        stacked_attns = ops.stack(attns, axis=1)

        # Zero out attentions at padding tokens
        mask = ops.cast(attention_mask, stacked_attns.dtype)
        m_row = mask[:, None, None, :, None]
        m_col = mask[:, None, None, None, :]
        stacked_attns = stacked_attns * m_row * m_col

        return self.contact_head(tokens, stacked_attns)

    def embed(
        self,
        sequences_or_tokens: Union[str, List[str], Any],
        pool: Optional[str] = "mean",
        tokenizer: Optional[Any] = None,
        return_numpy: bool = True,
    ) -> Any:
        """Extracts protein embeddings with optional pooling.

        Args:
            sequences_or_tokens: Protein sequence string, list of strings, or token IDs tensor.
            pool: Pooling mode: "mean" (average over residue tokens), "cls" (take <cls> token),
                or None / "none" (return per-residue embeddings).
            tokenizer: Optional ESMTokenizer instance. If None, instantiates ESMTokenizer.
            return_numpy: Whether to return numpy array (via ops.convert_to_numpy) or Keras tensor.

        Returns:
            Pooled embeddings of shape (batch, hidden_size) or per-residue embeddings
            of shape (batch, seq_len, hidden_size).
        """
        if isinstance(sequences_or_tokens, str):
            sequences = [sequences_or_tokens]
        elif (
            isinstance(sequences_or_tokens, list)
            and len(sequences_or_tokens) > 0
            and isinstance(sequences_or_tokens[0], str)
        ):
            sequences = sequences_or_tokens
        else:
            sequences = None

        if sequences is not None:
            if tokenizer is None:
                from keras_bio.protein.esm2.tokenizer import ESMTokenizer

                tokenizer = ESMTokenizer()
            encoded = tokenizer(sequences, return_tensors="np")
            input_ids = ops.convert_to_tensor(encoded["input_ids"])
            attention_mask = ops.convert_to_tensor(encoded["attention_mask"])
        else:
            input_ids, attention_mask, _ = _unpack_inputs(sequences_or_tokens)
            input_ids = ops.convert_to_tensor(input_ids)
            if attention_mask is not None:
                attention_mask = ops.convert_to_tensor(attention_mask)

        outputs = self(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            training=False,
        )
        hidden = outputs["last_hidden_state"]

        if pool == "cls":
            res = hidden[:, 0, :]
        elif pool == "mean":
            # Mask out padding, <cls>, and <eos>
            res_mask = ops.cast(ops.not_equal(input_ids, self.config.pad_token_id), "float32")
            res_mask = res_mask * ops.cast(ops.not_equal(input_ids, self.config.cls_token_id), "float32")
            res_mask = res_mask * ops.cast(ops.not_equal(input_ids, self.config.eos_token_id), "float32")
            res_mask_exp = ops.expand_dims(res_mask, -1)
            summed = ops.sum(hidden * res_mask_exp, axis=1)
            counts = ops.maximum(ops.sum(res_mask_exp, axis=1), 1.0)
            res = summed / counts
        else:
            res = hidden

        if return_numpy:
            return ops.convert_to_numpy(res)
        return res

    @classmethod
    def from_hf(
        cls,
        hf_id: str = "facebook/esm2_t6_8M_UR50D",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> "ESM2Model":
        """Load pretrained ESM-2 model from Hugging Face Hub or local directory."""
        from keras_bio.protein.esm2.convert import load_esm2_from_hf

        return load_esm2_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2Model":
        cfg = ESM2Config.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2ForMaskedLM(keras.Model):
    """ESM-2 model with Masked Language Modeling (MLM) head."""

    def __init__(self, config: Optional[ESM2Config] = None, **kwargs: Any):
        if config is None:
            valid = {f.name for f in ESM2Config.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = ESM2Config(**cfg_kwargs) if cfg_kwargs else ESM2Config()

        super().__init__(**kwargs)
        self.config = config
        self.esm = ESM2Model(config=config, name="esm")
        self.lm_head = ESM2LMHead(config=config, name="lm_head")

    def build(self, input_shape: Any) -> None:
        self.esm.build(input_shape)
        hidden_shape = (None, None, self.config.hidden_size)
        self.lm_head.build(hidden_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        esm_outputs = self.esm(
            inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = esm_outputs["last_hidden_state"]

        tied_weights = (
            self.esm.word_embeddings.embeddings
            if self.config.tie_word_embeddings
            else None
        )
        logits = self.lm_head(hidden_states, tied_embedding_weights=tied_weights)

        result = {"logits": logits, "last_hidden_state": hidden_states}
        if output_hidden_states:
            result["hidden_states"] = esm_outputs["hidden_states"]
        if output_attentions:
            result["attentions"] = esm_outputs["attentions"]
        return result

    @classmethod
    def from_hf(
        cls,
        hf_id: str = "facebook/esm2_t6_8M_UR50D",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> "ESM2ForMaskedLM":
        """Load pretrained ESM-2 MLM model from Hugging Face Hub."""
        from keras_bio.protein.esm2.convert import load_esm2_from_hf

        return load_esm2_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2ForMaskedLM":
        cfg = ESM2Config.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2ForSequenceClassification(keras.Model):
    """ESM-2 model with sequence-level classification head."""

    def __init__(
        self,
        config: Optional[ESM2Config] = None,
        num_labels: Optional[int] = None,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in ESM2Config.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = ESM2Config(**cfg_kwargs) if cfg_kwargs else ESM2Config()

        super().__init__(**kwargs)
        self.config = config
        self.num_labels = num_labels or config.num_labels
        self.esm = ESM2Model(config=config, name="esm")
        self.classifier = ESM2ClassificationHead(
            config=config, num_labels=self.num_labels, name="classifier"
        )

    def build(self, input_shape: Any) -> None:
        self.esm.build(input_shape)
        hidden_shape = (None, None, self.config.hidden_size)
        self.classifier.build(hidden_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        esm_outputs = self.esm(
            inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training,
        )
        logits = self.classifier(
            esm_outputs["last_hidden_state"], training=training
        )
        return {"logits": logits, "last_hidden_state": esm_outputs["last_hidden_state"]}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        config["num_labels"] = self.num_labels
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2ForSequenceClassification":
        cfg = ESM2Config.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class ESM2ForTokenClassification(keras.Model):
    """ESM-2 model with token/residue-level classification head."""

    def __init__(
        self,
        config: Optional[ESM2Config] = None,
        num_labels: Optional[int] = None,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in ESM2Config.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = ESM2Config(**cfg_kwargs) if cfg_kwargs else ESM2Config()

        super().__init__(**kwargs)
        self.config = config
        self.num_labels = num_labels or config.num_labels
        self.esm = ESM2Model(config=config, name="esm")
        self.dropout = layers.Dropout(config.hidden_dropout_prob)
        self.classifier = layers.Dense(self.num_labels, use_bias=True, name="classifier")

    def build(self, input_shape: Any) -> None:
        self.esm.build(input_shape)
        self.classifier.build((None, None, self.config.hidden_size))
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        esm_outputs = self.esm(
            inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training,
        )
        hidden = self.dropout(
            esm_outputs["last_hidden_state"], training=training
        )
        logits = self.classifier(hidden)
        return {"logits": logits, "last_hidden_state": esm_outputs["last_hidden_state"]}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        config["num_labels"] = self.num_labels
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ESM2ForTokenClassification":
        cfg = ESM2Config.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)
