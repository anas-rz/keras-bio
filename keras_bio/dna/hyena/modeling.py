"""HyenaDNA models for keras-bio.

Modular Keras 3 models supporting all backends (TensorFlow, PyTorch, JAX):
- HyenaDNAModel: Sub-quadratic genomic backbone with long convolution recurrences
- HyenaDNAForCausalLM: Causal language model for next-nucleotide generation
- HyenaDNAForSequenceClassification: Sequence classification/regression head
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import layers, ops
import numpy as np

from keras_bio.dna.hyena.config import HyenaDNAConfig
from keras_bio.dna.hyena.layers import HyenaBlock


def _unpack_hyena_inputs(inputs: Any) -> Tuple[Any, Optional[Any]]:
    """Extracts input_ids and optional attention_mask from inputs."""
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)
    elif isinstance(inputs, (list, tuple)):
        input_ids = inputs[0]
        attention_mask = inputs[1] if len(inputs) > 1 else None
    else:
        input_ids = inputs
        attention_mask = None
    return input_ids, attention_mask


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaDNAModel(keras.Model):
    """HyenaDNA sub-quadratic genomic backbone encoder.

    Args:
        config: Optional HyenaDNAConfig instance. Defaults to `hyenadna_tiny_1k`.
    """

    def __init__(
        self,
        config: Optional[HyenaDNAConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in HyenaDNAConfig.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = HyenaDNAConfig(**cfg_kwargs) if cfg_kwargs else HyenaDNAConfig()

        super().__init__(**kwargs)
        self.config = config

        self.word_embeddings = layers.Embedding(
            config.padded_vocab_size,
            config.d_model,
            name="word_embeddings",
        )
        self.embed_dropout = layers.Dropout(config.embed_dropout)

        self.encoder_layers = [
            HyenaBlock(config, name=f"layer_{i}") for i in range(config.n_layer)
        ]
        self.ln_f = layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_f"
        )

    def build(self, input_shape: Any) -> None:
        if isinstance(input_shape, dict):
            ids_shape = input_shape.get("input_ids", (None, None))
        elif isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
            ids_shape = input_shape[0]
        else:
            ids_shape = input_shape

        self.word_embeddings.build(ids_shape)
        embed_shape = (*ids_shape, self.config.d_model)
        for layer in self.encoder_layers:
            layer.build(embed_shape)
        self.ln_f.build(embed_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        input_ids, in_mask = _unpack_hyena_inputs(inputs)
        if attention_mask is None:
            attention_mask = in_mask

        hidden_states = self.word_embeddings(input_ids)
        hidden_states = self.embed_dropout(hidden_states, training=training)

        all_hidden_states = [] if output_hidden_states else None

        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, training=training)

        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        result: Dict[str, Any] = {"last_hidden_state": hidden_states}
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        return result

    def embed(
        self,
        sequences_or_tokens: Union[str, List[str], Any],
        pool: Optional[str] = "mean",
        tokenizer: Optional[Any] = None,
        return_numpy: bool = True,
    ) -> Any:
        """Extracts genomic embeddings with optional pooling.

        Args:
            sequences_or_tokens: DNA string, list of strings, or token IDs tensor.
            pool: Pooling mode: "mean" (average over non-padding tokens), "last"
                (causal last non-padding token), or None / "none" (per-nucleotide).
            tokenizer: Optional HyenaDNATokenizer instance. If None, instantiates one.
            return_numpy: Whether to return numpy array via ops.convert_to_numpy.

        Returns:
            Pooled embeddings of shape (batch, d_model) or per-nucleotide embeddings
            of shape (batch, seq_len, d_model).
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
                from keras_bio.dna.hyena.tokenizer import HyenaDNATokenizer

                tokenizer = HyenaDNATokenizer()
            encoded = tokenizer(sequences, return_tensors="np")
            input_ids = ops.convert_to_tensor(encoded["input_ids"])
            attention_mask = ops.convert_to_tensor(encoded["attention_mask"])
        else:
            input_ids, attention_mask = _unpack_hyena_inputs(sequences_or_tokens)
            input_ids = ops.convert_to_tensor(input_ids)
            if attention_mask is not None:
                attention_mask = ops.convert_to_tensor(attention_mask)

        outputs = self({"input_ids": input_ids, "attention_mask": attention_mask}, training=False)
        hidden = outputs["last_hidden_state"]

        if pool == "last":
            # Select last non-pad token for each sequence
            non_pad = ops.cast(ops.not_equal(input_ids, self.config.pad_token_id), "int32")
            lengths = ops.maximum(ops.sum(non_pad, axis=-1) - 1, 0)
            seq_len = ops.shape(hidden)[1]
            one_hot = ops.one_hot(lengths, num_classes=seq_len)
            one_hot_exp = ops.expand_dims(ops.cast(one_hot, hidden.dtype), -1)
            res = ops.sum(hidden * one_hot_exp, axis=1)
        elif pool == "mean":
            # Mask out padding tokens
            mask = ops.cast(ops.not_equal(input_ids, self.config.pad_token_id), "float32")
            mask_exp = ops.expand_dims(mask, -1)
            summed = ops.sum(hidden * mask_exp, axis=1)
            counts = ops.maximum(ops.sum(mask_exp, axis=1), 1.0)
            res = summed / counts
        else:
            res = hidden

        if return_numpy:
            return ops.convert_to_numpy(res)
        return res

    @classmethod
    def from_hf(
        cls,
        hf_id: str = "LongSafari/hyenadna-tiny-1k-seqlen-hf",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> "HyenaDNAModel":
        """Load pretrained HyenaDNA model from Hugging Face Hub or local directory."""
        from keras_bio.dna.hyena.convert import load_hyenadna_from_hf

        return load_hyenadna_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaDNAModel":
        cfg = HyenaDNAConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaDNAForCausalLM(keras.Model):
    """HyenaDNA model with causal language modeling head for autoregressive generation."""

    def __init__(
        self,
        config: Optional[HyenaDNAConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in HyenaDNAConfig.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = HyenaDNAConfig(**cfg_kwargs) if cfg_kwargs else HyenaDNAConfig()

        super().__init__(**kwargs)
        self.config = config
        self.hyena = HyenaDNAModel(config=config, name="hyena")
        self.lm_head = layers.Dense(
            config.padded_vocab_size,
            use_bias=False,
            name="lm_head",
        )

    def build(self, input_shape: Any) -> None:
        self.hyena.build(input_shape)
        hidden_shape = (None, None, self.config.d_model)
        self.lm_head.build(hidden_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        hyena_outputs = self.hyena(
            inputs,
            attention_mask=attention_mask,
            training=training,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = hyena_outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        result: Dict[str, Any] = {
            "logits": logits,
            "last_hidden_state": hidden_states,
        }
        if output_hidden_states:
            result["hidden_states"] = hyena_outputs["hidden_states"]
        return result

    @classmethod
    def from_hf(
        cls,
        hf_id: str = "LongSafari/hyenadna-tiny-1k-seqlen-hf",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> "HyenaDNAForCausalLM":
        """Load pretrained HyenaDNA causal LM from Hugging Face Hub."""
        from keras_bio.dna.hyena.convert import load_hyenadna_from_hf

        return load_hyenadna_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaDNAForCausalLM":
        cfg = HyenaDNAConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaDNAForSequenceClassification(keras.Model):
    """HyenaDNA model with sequence classification head at the last non-pad token."""

    def __init__(
        self,
        config: Optional[HyenaDNAConfig] = None,
        num_labels: Optional[int] = None,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in HyenaDNAConfig.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = HyenaDNAConfig(**cfg_kwargs) if cfg_kwargs else HyenaDNAConfig()

        super().__init__(**kwargs)
        self.config = config
        self.num_labels = num_labels or config.num_labels
        self.hyena = HyenaDNAModel(config=config, name="hyena")
        self.score = layers.Dense(self.num_labels, use_bias=False, name="score")

    def build(self, input_shape: Any) -> None:
        self.hyena.build(input_shape)
        cls_shape = (None, self.config.d_model)
        self.score.build(cls_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        input_ids, in_mask = _unpack_hyena_inputs(inputs)
        if attention_mask is None:
            attention_mask = in_mask

        hyena_outputs = self.hyena(
            inputs,
            attention_mask=attention_mask,
            training=training,
        )
        hidden = hyena_outputs["last_hidden_state"]

        # Pool at the last non-pad token
        non_pad = ops.cast(ops.not_equal(input_ids, self.config.pad_token_id), "int32")
        sequence_lengths = ops.maximum(ops.sum(non_pad, axis=-1) - 1, 0)
        seq_len = ops.shape(hidden)[1]
        one_hot = ops.one_hot(sequence_lengths, num_classes=seq_len)
        one_hot_exp = ops.expand_dims(ops.cast(one_hot, hidden.dtype), -1)
        pooled = ops.sum(hidden * one_hot_exp, axis=1)

        logits = self.score(pooled)
        return {
            "logits": logits,
            "last_hidden_state": hidden,
            "pooled_hidden_state": pooled,
        }

    @classmethod
    def from_hf(
        cls,
        hf_id: str = "LongSafari/hyenadna-tiny-1k-seqlen-hf",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> "HyenaDNAForSequenceClassification":
        """Load pretrained HyenaDNA sequence classifier from Hugging Face Hub."""
        from keras_bio.dna.hyena.convert import load_hyenadna_from_hf

        return load_hyenadna_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        config["num_labels"] = self.num_labels
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaDNAForSequenceClassification":
        cfg = HyenaDNAConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)
