"""ProtT5 models for keras-bio.

Modular Keras 3 models supporting all backends (TensorFlow, PyTorch, JAX):
- ProtT5EncoderModel: Encoder-only backbone for residue and protein embeddings
- ProtT5Model: Full encoder-decoder backbone
- ProtT5ForConditionalGeneration: Full model with language modeling head and generate()
- ProtT5ForSequenceClassification: Protein sequence classifier (e.g. localization, function)
- ProtT5ForTokenClassification: Per-residue classifier (e.g. secondary structure, binding sites)
"""

from typing import Any, Dict, List, Optional, Union

import keras
from keras import layers, ops
import numpy as np

from keras_bio.protein.prot_t5.config import ProtT5Config
from keras_bio.protein.prot_t5.layers import (
    MASK_NEG,
    ProtT5DecoderBlock,
    ProtT5EncoderBlock,
    ProtT5LayerNorm,
    ProtT5PositionBias,
    ProtT5TiedHead,
)


def _build_encoder(cfg: ProtT5Config):
    """Create encoder layers from ProtT5Config."""
    shared = layers.Embedding(cfg.vocab_size, cfg.d_model, name="shared")
    rel_bias = layers.Embedding(
        cfg.relative_attention_num_buckets, cfg.num_heads, name="encoder_rel_bias"
    )
    bias_layer = ProtT5PositionBias(
        num_heads=cfg.num_heads,
        num_buckets=cfg.relative_attention_num_buckets,
        max_distance=cfg.relative_attention_max_distance,
        bidirectional=True,
        causal=False,
        rel_bias_embedding=rel_bias,
        name="encoder_bias",
    )
    blocks = [
        ProtT5EncoderBlock(
            embed_dim=cfg.d_model,
            key_value_dim=cfg.d_kv,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.d_ff,
            hidden_act=cfg.dense_act_fn,
            is_gated_act=cfg.is_gated_act,
            eps=cfg.layer_norm_epsilon,
            dropout=cfg.dropout_rate,
            prefix=f"enc_{i}",
            name=f"encoder_block_{i}",
        )
        for i in range(cfg.num_layers)
    ]
    final_norm = ProtT5LayerNorm(
        cfg.layer_norm_epsilon, name="encoder_final_layer_norm"
    )
    dropout_layer = layers.Dropout(cfg.dropout_rate, name="encoder_dropout")
    return shared, rel_bias, bias_layer, blocks, final_norm, dropout_layer


def _build_decoder(cfg: ProtT5Config):
    """Create decoder layers from ProtT5Config."""
    rel_bias = layers.Embedding(
        cfg.relative_attention_num_buckets, cfg.num_heads, name="decoder_rel_bias"
    )
    bias_layer = ProtT5PositionBias(
        num_heads=cfg.num_heads,
        num_buckets=cfg.relative_attention_num_buckets,
        max_distance=cfg.relative_attention_max_distance,
        bidirectional=False,
        causal=True,
        rel_bias_embedding=rel_bias,
        name="decoder_bias",
    )
    num_dec = cfg.num_decoder_layers or cfg.num_layers
    blocks = [
        ProtT5DecoderBlock(
            embed_dim=cfg.d_model,
            key_value_dim=cfg.d_kv,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.d_ff,
            hidden_act=cfg.dense_act_fn,
            is_gated_act=cfg.is_gated_act,
            eps=cfg.layer_norm_epsilon,
            dropout=cfg.dropout_rate,
            prefix=f"dec_{i}",
            name=f"decoder_block_{i}",
        )
        for i in range(num_dec)
    ]
    final_norm = ProtT5LayerNorm(
        cfg.layer_norm_epsilon, name="decoder_final_layer_norm"
    )
    dropout_layer = layers.Dropout(cfg.dropout_rate, name="decoder_dropout")
    return rel_bias, bias_layer, blocks, final_norm, dropout_layer


def _shift_right(input_ids, decoder_start_token_id: int):
    """Graph-safe right shift: prepend decoder start token and drop last token."""
    start = ops.full_like(input_ids[:, :1], decoder_start_token_id)
    return ops.concatenate([start, input_ids[:, :-1]], axis=1)


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5EncoderModel(keras.Model):
    """ProtT5 Transformer Encoder model (no decoder, no LM head).

    Takes protein token IDs and outputs per-residue embeddings `(batch, seq, d_model)`.
    Primary model used for protein representation learning, variant effect prediction,
    and downstream feature extraction.
    """

    def __init__(self, config: Optional[ProtT5Config] = None, **kwargs):
        if config is None:
            # Extract config parameters from kwargs if provided
            valid = {f.name for f in ProtT5Config.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = ProtT5Config(**cfg_kwargs) if cfg_kwargs else ProtT5Config()

        super().__init__(**kwargs)
        self.config = config

        (
            self.shared,
            self.encoder_rel_bias,
            self.encoder_bias,
            self.encoder_blocks,
            self.encoder_final_layer_norm,
            self.encoder_dropout,
        ) = _build_encoder(config)

    def build(self, input_shape=None):
        dummy_dim = (None, None)
        self.shared.build((None, None))
        self.encoder_rel_bias.build((self.config.relative_attention_num_buckets,))
        embed_shape = (None, None, self.config.d_model)
        for block in self.encoder_blocks:
            block.build(embed_shape)
        self.encoder_final_layer_norm.build(embed_shape)
        super().build(input_shape or dummy_dim)

    def encode(
        self,
        input_ids,
        attention_mask=None,
        training: Optional[bool] = None,
    ):
        """Run the encoder body."""
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        hidden = self.shared(input_ids)
        hidden = self.encoder_dropout(hidden, training=training)

        bias = self.encoder_bias(hidden, attention_mask=attention_mask)
        for block in self.encoder_blocks:
            hidden = block(hidden, position_bias=bias, training=training)
        return self.encoder_final_layer_norm(hidden)

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        training: Optional[bool] = None,
        **kwargs,
    ):
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask", attention_mask)
            input_ids = input_ids["input_ids"]
        hidden = self.encode(input_ids, attention_mask=attention_mask, training=training)
        return {"last_hidden_state": hidden}

    def embed(
        self,
        sequences: Union[str, List[str]],
        tokenizer: Optional[Any] = None,
        pool: str = "mean",
    ):
        """Convenience method to compute embeddings directly from sequence strings.

        Args:
            sequences: Single protein sequence or list of sequence strings.
            tokenizer: Tokenizer or HuggingFace T5Tokenizer.
            pool: "mean" for per-protein vector, "none" for per-residue tensor.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if tokenizer is None:
            from keras_bio.protein.prot_t5.tokenizer import ProtT5Tokenizer

            tokenizer = ProtT5Tokenizer()

        encoded = tokenizer(sequences, return_tensors="np")
        inputs = {
            "input_ids": ops.convert_to_tensor(encoded["input_ids"]),
            "attention_mask": ops.convert_to_tensor(encoded["attention_mask"]),
        }
        outputs = self(inputs, training=False)
        hidden = outputs["last_hidden_state"]

        if pool == "mean":
            mask = ops.cast(inputs["attention_mask"][:, :, None], hidden.dtype)
            summed = ops.sum(hidden * mask, axis=1)
            lengths = ops.maximum(ops.sum(mask, axis=1), 1.0)
            return summed / lengths
        return hidden

    @classmethod
    def from_hf(
        cls, hf_id: str = "Rostlab/prot_t5_xl_half_uniref50-enc", dtype: str = "float32", **kwargs
    ) -> "ProtT5EncoderModel":
        """Load pretrained model from Hugging Face Hub or local directory."""
        from keras_bio.protein.prot_t5.convert import load_t5_encoder_from_hf

        return load_t5_encoder_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config):
        cfg = ProtT5Config.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5Model(keras.Model):
    """Full ProtT5 Transformer Encoder-Decoder model (without language modeling head)."""

    def __init__(self, config: Optional[ProtT5Config] = None, **kwargs):
        if config is None:
            valid = {f.name for f in ProtT5Config.__dataclass_fields__.values()}
            cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
            config = ProtT5Config(**cfg_kwargs) if cfg_kwargs else ProtT5Config()

        super().__init__(**kwargs)
        self.config = config

        (
            self.shared,
            self.encoder_rel_bias,
            self.encoder_bias,
            self.encoder_blocks,
            self.encoder_final_layer_norm,
            self.encoder_dropout,
        ) = _build_encoder(config)

        (
            self.decoder_rel_bias,
            self.decoder_bias,
            self.decoder_blocks,
            self.decoder_final_layer_norm,
            self.decoder_dropout,
        ) = _build_decoder(config)

    def build(self, input_shape=None):
        dummy_dim = (None, None)
        self.shared.build((None, None))
        self.encoder_rel_bias.build((self.config.relative_attention_num_buckets,))
        self.decoder_rel_bias.build((self.config.relative_attention_num_buckets,))
        embed_shape = (None, None, self.config.d_model)
        for block in self.encoder_blocks:
            block.build(embed_shape)
        for block in self.decoder_blocks:
            block.build(embed_shape)
        self.encoder_final_layer_norm.build(embed_shape)
        self.decoder_final_layer_norm.build(embed_shape)
        super().build(input_shape or dummy_dim)

    def encode(
        self,
        input_ids,
        attention_mask=None,
        training: Optional[bool] = None,
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        hidden = self.shared(input_ids)
        hidden = self.encoder_dropout(hidden, training=training)

        bias = self.encoder_bias(hidden, attention_mask=attention_mask)
        for block in self.encoder_blocks:
            hidden = block(hidden, position_bias=bias, training=training)
        return self.encoder_final_layer_norm(hidden)

    def decode(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        training: Optional[bool] = None,
    ):
        decoder_input_ids = ops.cast(ops.convert_to_tensor(decoder_input_ids), "int32")
        hidden = self.shared(decoder_input_ids)
        hidden = self.decoder_dropout(hidden, training=training)

        self_bias = self.decoder_bias(hidden, attention_mask=decoder_attention_mask)
        if encoder_attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(encoder_attention_mask), "float32")
            cross_bias = (1.0 - am)[:, None, None, :] * MASK_NEG
        else:
            cross_bias = None

        for block in self.decoder_blocks:
            hidden = block(
                hidden,
                self_position_bias=self_bias,
                encoder_hidden_states=encoder_hidden_states,
                cross_position_bias=cross_bias,
                training=training,
            )
        return self.decoder_final_layer_norm(hidden)

    def call(
        self,
        input_ids=None,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        training: Optional[bool] = None,
        **kwargs,
    ):
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask", attention_mask)
            decoder_input_ids = input_ids.get("decoder_input_ids", decoder_input_ids)
            decoder_attention_mask = input_ids.get(
                "decoder_attention_mask", decoder_attention_mask
            )
            input_ids = input_ids["input_ids"]

        if decoder_input_ids is None:
            decoder_input_ids = _shift_right(
                input_ids, self.config.decoder_start_token_id
            )

        encoder_hidden = self.encode(
            input_ids, attention_mask=attention_mask, training=training
        )
        decoder_hidden = self.decode(
            decoder_input_ids,
            encoder_hidden,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            training=training,
        )
        return {
            "last_hidden_state": decoder_hidden,
            "encoder_last_hidden_state": encoder_hidden,
        }

    @classmethod
    def from_hf(
        cls, hf_id: str, dtype: str = "float32", **kwargs
    ) -> "ProtT5Model":
        from keras_bio.protein.prot_t5.convert import load_t5_model_from_hf

        return load_t5_model_from_hf(cls, hf_id, dtype=dtype, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config):
        cfg = ProtT5Config.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        return cls(config=cfg, **kwargs)


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5ForConditionalGeneration(ProtT5Model):
    """ProtT5 encoder-decoder model with a language modeling head and text generation."""

    def __init__(self, config: Optional[ProtT5Config] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        if self.config.tie_word_embeddings:
            self.tied_head = ProtT5TiedHead(name="lm_head")
            self.lm_head = None
        else:
            self.tied_head = None
            self.lm_head = layers.Dense(
                self.config.vocab_size, use_bias=False, name="lm_head"
            )

    def project_logits(self, decoder_hidden):
        """Compute vocabulary logits from decoder hidden states."""
        if self.config.tie_word_embeddings:
            scaled = decoder_hidden * (self.config.d_model**-0.5)
            return self.tied_head([scaled, self.shared.embeddings])
        return self.lm_head(decoder_hidden)

    def call(
        self,
        input_ids=None,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        training: Optional[bool] = None,
        **kwargs,
    ):
        base_out = super().call(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            training=training,
            **kwargs,
        )
        logits = self.project_logits(base_out["last_hidden_state"])
        base_out["logits"] = logits
        return base_out

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
    ):
        """Autoregressive greedy sequence generation."""
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        if attention_mask is not None:
            attention_mask = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        batch = int(ops.shape(input_ids)[0])

        encoder_hidden = self.encode(input_ids, attention_mask=attention_mask)
        generated = ops.full(
            (batch, 1), self.config.decoder_start_token_id, dtype="int32"
        )
        done = ops.zeros((batch,), dtype="bool")

        for _ in range(max_new_tokens):
            decoder_hidden = self.decode(
                generated,
                encoder_hidden,
                decoder_attention_mask=None,
                encoder_attention_mask=attention_mask,
            )
            next_logits = self.project_logits(decoder_hidden)[:, -1, :]
            next_ids = ops.cast(ops.argmax(next_logits, axis=-1), "int32")
            next_ids = ops.cast(ops.where(done, eos, next_ids), "int32")
            generated = ops.concatenate([generated, next_ids[:, None]], axis=1)
            done = ops.logical_or(done, ops.equal(next_ids, eos))
            if bool(ops.all(done)):
                break
        return generated


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5ForSequenceClassification(ProtT5Model):
    """ProtT5 model with a sequence classification head.

    Decodes right-shifted input_ids, pools the decoder hidden state at the last EOS token,
    and projects to classification logits.
    """

    def __init__(
        self,
        config: Optional[ProtT5Config] = None,
        num_classes: int = 2,
        classifier_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.num_classes = num_classes
        self.classifier_dropout_rate = classifier_dropout
        self.classifier_dense = layers.Dense(
            self.config.d_model, activation="tanh", name="classification_head_dense"
        )
        self.classifier_dropout = layers.Dropout(
            classifier_dropout, name="classification_head_dropout"
        )
        self.classifier_out_proj = layers.Dense(
            num_classes, name="classification_head_out_proj"
        )

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        training: Optional[bool] = None,
        **kwargs,
    ):
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask", attention_mask)
            input_ids = input_ids["input_ids"]

        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        decoder_input_ids = _shift_right(
            input_ids, self.config.decoder_start_token_id
        )

        base_out = super().call(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            training=training,
        )
        decoder_hidden = base_out["last_hidden_state"]

        # Pool at the last EOS token position (matching Hugging Face T5ForSequenceClassification)
        seq = ops.shape(input_ids)[1]
        eos_positions = ops.where(
            ops.equal(input_ids, self.config.eos_token_id),
            ops.arange(seq)[None, :],
            -1,
        )
        last_eos = ops.max(eos_positions, axis=1)
        pooled = ops.take_along_axis(
            decoder_hidden, last_eos[:, None, None], axis=1
        )[:, 0, :]

        pooled = self.classifier_dropout(pooled, training=training)
        hidden = self.classifier_dense(pooled)
        hidden = self.classifier_dropout(hidden, training=training)
        logits = self.classifier_out_proj(hidden)
        return {"logits": logits, "pooled": pooled}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_bio")
class ProtT5ForTokenClassification(ProtT5EncoderModel):
    """ProtT5 encoder model with a per-token classification head.

    Outputs logits of shape `(batch, seq, num_classes)`. Suitable for residue-level
    predictions such as secondary structure, disorder, and active sites.
    """

    def __init__(
        self,
        config: Optional[ProtT5Config] = None,
        num_classes: int = 3,
        classifier_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.num_classes = num_classes
        self.classifier_dropout_rate = classifier_dropout
        self.classifier_dropout = layers.Dropout(
            classifier_dropout, name="classifier_dropout"
        )
        self.classifier = layers.Dense(num_classes, name="classifier")

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        training: Optional[bool] = None,
        **kwargs,
    ):
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask", attention_mask)
            input_ids = input_ids["input_ids"]

        base_out = super().call(
            input_ids=input_ids, attention_mask=attention_mask, training=training
        )
        hidden = self.classifier_dropout(
            base_out["last_hidden_state"], training=training
        )
        logits = self.classifier(hidden)
        return {"logits": logits, "last_hidden_state": base_out["last_hidden_state"]}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout_rate,
            }
        )
        return config


# Backward-compatible aliases
T5EncoderModel = ProtT5EncoderModel
T5Model = ProtT5Model
T5ConditionalGenerate = ProtT5ForConditionalGeneration
T5ForConditionalGeneration = ProtT5ForConditionalGeneration
T5SequenceClassify = ProtT5ForSequenceClassification
T5TokenClassify = ProtT5ForTokenClassification

