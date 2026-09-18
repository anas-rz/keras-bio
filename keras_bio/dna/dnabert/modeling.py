"""DNABERT models for keras-bio.

Modular Keras 3 models supporting all backends (TensorFlow, PyTorch, JAX):
- DNABERTModel: BERT Transformer encoder backbone with pooler
- DNABERTForMaskedLM: Masked language modeling for zero-shot genomic variant effect prediction
- DNABERTForSequenceClassification: Sequence-level classification head (promoters, enhancers, splice sites)
- DNABERTForTokenClassification: Token-level classification head (transcription factor binding sites)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import layers, ops

from keras_bio.dna.dnabert.config import DNABERTConfig
from keras_bio.dna.dnabert.layers import (
    DNABERTEmbeddings,
    DNABERTLMHead,
    DNABERTLayer,
    DNABERTPooler,
)


def _unpack_dnabert_inputs(inputs: Any) -> Tuple[Any, Any, Any, Any]:
    """Extracts input_ids, attention_mask, token_type_ids, position_ids from inputs."""
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)
        position_ids = inputs.get("position_ids", None)
    elif isinstance(inputs, (list, tuple)):
        input_ids = inputs[0]
        attention_mask = inputs[1] if len(inputs) > 1 else None
        token_type_ids = inputs[2] if len(inputs) > 2 else None
        position_ids = inputs[3] if len(inputs) > 3 else None
    else:
        input_ids = inputs
        attention_mask = None
        token_type_ids = None
        position_ids = None
    return input_ids, attention_mask, token_type_ids, position_ids


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTModel(keras.Model):
    """DNABERT Transformer Encoder backbone with pooler.

    Args:
        config: Optional DNABERTConfig instance. If not provided, defaults to
            6-mer DNABERT preset (`dnabert_6`).
        add_pooling_layer: Whether to include the pooler layer. Defaults to True.
    """

    def __init__(
        self,
        config: Optional[DNABERTConfig] = None,
        add_pooling_layer: bool = True,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in DNABERTConfig.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = DNABERTConfig(**cfg_kwargs) if cfg_kwargs else DNABERTConfig()

        super().__init__(**kwargs)
        self.config = config
        self.add_pooling_layer = add_pooling_layer

        self.embeddings = DNABERTEmbeddings(config, name="embeddings")
        self.encoder_layers = [
            DNABERTLayer(config, name=f"layer_{i}")
            for i in range(config.num_hidden_layers)
        ]
        if add_pooling_layer:
            self.pooler = DNABERTPooler(config, name="pooler")
        else:
            self.pooler = None

    def build(self, input_shape: Any) -> None:
        self.embeddings.build(input_shape)
        if isinstance(input_shape, dict):
            ids_shape = input_shape.get("input_ids", (None, None))
        elif isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
            ids_shape = input_shape[0]
        else:
            ids_shape = input_shape

        embed_shape = (*ids_shape, self.config.hidden_size)
        for layer in self.encoder_layers:
            layer.build(embed_shape)

        if self.pooler is not None:
            self.pooler.build(embed_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        input_ids, in_mask, in_types, in_pos = _unpack_dnabert_inputs(inputs)
        if attention_mask is None:
            attention_mask = in_mask
        if token_type_ids is None:
            token_type_ids = in_types
        if position_ids is None:
            position_ids = in_pos

        hidden_states = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training,
        )

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if output_attentions:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    training=training,
                    output_attentions=True,
                )
                all_attentions.append(attn_weights)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    training=training,
                    output_attentions=False,
                )

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        result: Dict[str, Any] = {"last_hidden_state": hidden_states}
        if pooled_output is not None:
            result["pooler_output"] = pooled_output
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        if output_attentions:
            result["attentions"] = all_attentions
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
            sequences_or_tokens: DNA sequence string, list of strings, or token IDs tensor.
            pool: Pooling mode: "mean" (average over k-mer tokens), "cls" (take pooled [CLS]),
                or None / "none" (return per-kmer embeddings).
            tokenizer: Optional DNABERTTokenizer instance. If None, instantiates DNABERTTokenizer.
            return_numpy: Whether to return numpy array (via ops.convert_to_numpy) or Keras tensor.

        Returns:
            Pooled embeddings of shape (batch, hidden_size) or per-kmer embeddings
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
                from keras_bio.dna.dnabert.tokenizer import DNABERTTokenizer

                tokenizer = DNABERTTokenizer(kmer=self.config.kmer)
            encoded = tokenizer(sequences, return_tensors="np")
            input_ids = ops.convert_to_tensor(encoded["input_ids"])
            attention_mask = ops.convert_to_tensor(encoded["attention_mask"])
        else:
            input_ids, attention_mask, _, _ = _unpack_dnabert_inputs(sequences_or_tokens)
            input_ids = ops.convert_to_tensor(input_ids)
            if attention_mask is not None:
                attention_mask = ops.convert_to_tensor(attention_mask)

        outputs = self(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            training=False,
        )
        hidden = outputs["last_hidden_state"]

        if pool == "cls":
            if "pooler_output" in outputs:
                res = outputs["pooler_output"]
            else:
                res = hidden[:, 0, :]
        elif pool == "mean":
            # Mask out padding, [CLS], and [SEP]
            res_mask = ops.cast(ops.not_equal(input_ids, self.config.pad_token_id), "float32")
            res_mask = res_mask * ops.cast(ops.not_equal(input_ids, self.config.cls_token_id), "float32")
            res_mask = res_mask * ops.cast(ops.not_equal(input_ids, self.config.sep_token_id), "float32")
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
        hf_id: str = "zhihan1996/DNA_bert_6",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> "DNABERTModel":
        """Load pretrained DNABERT model from Hugging Face Hub or local directory."""
        from keras_bio.dna.dnabert.convert import load_dnabert_from_hf

        return load_dnabert_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        config["add_pooling_layer"] = self.add_pooling_layer
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTModel":
        pooling = config.pop("add_pooling_layer", True)
        cfg = DNABERTConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, add_pooling_layer=pooling, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTForMaskedLM(keras.Model):
    """DNABERT model with Masked Language Modeling (MLM) head."""

    def __init__(self, config: Optional[DNABERTConfig] = None, **kwargs: Any):
        if config is None:
            valid = {f.name for f in DNABERTConfig.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = DNABERTConfig(**cfg_kwargs) if cfg_kwargs else DNABERTConfig()

        super().__init__(**kwargs)
        self.config = config
        self.bert = DNABERTModel(config=config, add_pooling_layer=False, name="bert")
        self.lm_head = DNABERTLMHead(config=config, name="lm_head")

    def build(self, input_shape: Any) -> None:
        self.bert.build(input_shape)
        hidden_shape = (None, None, self.config.hidden_size)
        self.lm_head.build(hidden_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        bert_outputs = self.bert(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = bert_outputs["last_hidden_state"]

        tied_weights = (
            self.bert.embeddings.word_embeddings.embeddings
            if self.config.tie_word_embeddings
            else None
        )
        logits = self.lm_head(hidden_states, tied_embedding_weights=tied_weights)

        result = {"logits": logits, "last_hidden_state": hidden_states}
        if output_hidden_states:
            result["hidden_states"] = bert_outputs["hidden_states"]
        if output_attentions:
            result["attentions"] = bert_outputs["attentions"]
        return result

    @classmethod
    def from_hf(
        cls,
        hf_id: str = "zhihan1996/DNA_bert_6",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> "DNABERTForMaskedLM":
        """Load pretrained DNABERT MLM model from Hugging Face Hub."""
        from keras_bio.dna.dnabert.convert import load_dnabert_from_hf

        return load_dnabert_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTForMaskedLM":
        cfg = DNABERTConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTForSequenceClassification(keras.Model):
    """DNABERT model with sequence-level classification head on pooled [CLS] embedding."""

    def __init__(
        self,
        config: Optional[DNABERTConfig] = None,
        num_labels: Optional[int] = None,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in DNABERTConfig.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = DNABERTConfig(**cfg_kwargs) if cfg_kwargs else DNABERTConfig()

        super().__init__(**kwargs)
        self.config = config
        self.num_labels = num_labels or config.num_labels
        self.bert = DNABERTModel(config=config, add_pooling_layer=True, name="bert")
        dropout_prob = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = layers.Dropout(dropout_prob)
        self.classifier = layers.Dense(self.num_labels, use_bias=True, name="classifier")

    def build(self, input_shape: Any) -> None:
        self.bert.build(input_shape)
        cls_shape = (None, self.config.hidden_size)
        self.classifier.build(cls_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        bert_outputs = self.bert(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training,
        )
        pooled_output = bert_outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return {
            "logits": logits,
            "last_hidden_state": bert_outputs["last_hidden_state"],
            "pooler_output": pooled_output,
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        config["num_labels"] = self.num_labels
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTForSequenceClassification":
        cfg = DNABERTConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class DNABERTForTokenClassification(keras.Model):
    """DNABERT model with per-token/per-kmer classification head."""

    def __init__(
        self,
        config: Optional[DNABERTConfig] = None,
        num_labels: Optional[int] = None,
        **kwargs: Any,
    ):
        if config is None:
            valid = {f.name for f in DNABERTConfig.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = DNABERTConfig(**cfg_kwargs) if cfg_kwargs else DNABERTConfig()

        super().__init__(**kwargs)
        self.config = config
        self.num_labels = num_labels or config.num_labels
        self.bert = DNABERTModel(config=config, add_pooling_layer=False, name="bert")
        dropout_prob = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = layers.Dropout(dropout_prob)
        self.classifier = layers.Dense(self.num_labels, use_bias=True, name="classifier")

    def build(self, input_shape: Any) -> None:
        self.bert.build(input_shape)
        self.classifier.build((None, None, self.config.hidden_size))
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        bert_outputs = self.bert(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training,
        )
        hidden = self.dropout(bert_outputs["last_hidden_state"], training=training)
        logits = self.classifier(hidden)
        return {
            "logits": logits,
            "last_hidden_state": bert_outputs["last_hidden_state"],
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        config["num_labels"] = self.num_labels
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNABERTForTokenClassification":
        cfg = DNABERTConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)

