"""Keras 3 layers for HyenaDNA models.

Implements the sub-quadratic Hyena recurrence operator with:
- `fftconv`: Causal 1D convolution via real Fast Fourier Transforms
- `HyenaSin`: Trainable sine activation function
- `HyenaPositionalEmbedding`: Multi-band cosine and sine positional encoding
- `HyenaExponentialModulation`: Exponential window function for filter decay
- `HyenaFilter`: Implicit MLP predicting long convolution kernels
- `HyenaOperator`: Gated implicit convolution recurrence block
- `HyenaMlp`: Feed-forward network with tanh-approximated GELU
- `HyenaBlock`: Pre-LayerNorm Transformer block with Hyena recurrence and MLP
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import keras
from keras import layers, ops
import numpy as np

from keras_bio.dna.hyena.config import HyenaDNAConfig


def fftconv(
    u: keras.KerasTensor,
    k: keras.KerasTensor,
    D: keras.KerasTensor,
) -> keras.KerasTensor:
    """Causal 1D convolution through the Fourier domain using the Convolution Theorem.

    Computes: y = irfft(rfft(u) * rfft(k))[..., :seqlen] + u * D

    Args:
        u: Input tensor of shape (batch, channels, seqlen).
        k: Convolution kernel of shape (channels, seqlen).
        D: Direct feedthrough bias of shape (channels,).

    Returns:
        Convolved output tensor of shape (batch, channels, seqlen).
    """
    seqlen = ops.shape(u)[-1]
    fft_size = 2 * seqlen

    u_real, u_imag = ops.rfft(u, fft_length=fft_size)
    k_real, k_imag = ops.rfft(k, fft_length=fft_size)

    # Complex multiplication: (u_r + i*u_i) * (k_r + i*k_i)
    prod_real = u_real * k_real - u_imag * k_imag
    prod_imag = u_real * k_imag + u_imag * k_real

    # Inverse RFFT with standard 1/fft_size scaling
    y = ops.irfft((prod_real, prod_imag), fft_length=fft_size)[..., :seqlen]

    # Direct feedthrough connection
    out = y + u * ops.expand_dims(D, -1)
    return ops.cast(out, u.dtype)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaSin(layers.Layer):
    """Sine activation layer with trainable frequency multiplier."""

    def __init__(
        self,
        config: HyenaDNAConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.activation_freq = float(config.activation_freq)
        self.filter_order = int(config.filter_order)
        self.train_freq = bool(config.train_freq)

    def build(self, input_shape: Any) -> None:
        self.freq = self.add_weight(
            name="freq",
            shape=(1, self.filter_order),
            initializer=keras.initializers.Constant(self.activation_freq),
            trainable=self.train_freq,
        )
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        return ops.sin(self.freq * x)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"config": self.config.to_dict()})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaSin":
        cfg = HyenaDNAConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaPositionalEmbedding(layers.Layer):
    """Multi-band complex exponential positional embeddings for Hyena filters."""

    def __init__(
        self,
        config: HyenaDNAConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.seq_len = int(config.max_seq_len)
        self.emb_dim = int(config.emb_dim)

    def build(self, input_shape: Any = None) -> None:
        # Precompute normalized time: [1, seq_len, 1]
        t_np = np.linspace(0.0, 1.0, self.seq_len, dtype="float32")[None, :, None]

        bands = (self.emb_dim - 1) // 2
        t_rescaled = np.linspace(0.0, self.seq_len - 1.0, self.seq_len, dtype="float32")[
            None, :, None
        ]
        w = 2.0 * math.pi * t_rescaled / self.seq_len
        f = np.linspace(1e-4, bands - 1.0, bands, dtype="float32")[None, None, :]

        cos_term = np.cos(-f * w)
        sin_term = np.sin(-f * w)
        z_np = np.concatenate([t_np, cos_term, sin_term], axis=-1)

        self.z = self.add_weight(
            name="z",
            shape=(1, self.seq_len, self.emb_dim),
            initializer=keras.initializers.Constant(z_np),
            trainable=False,
        )
        self.t = self.add_weight(
            name="t",
            shape=(1, self.seq_len, 1),
            initializer=keras.initializers.Constant(t_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, L: Any) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        return self.z[:, :L], self.t[:, :L]

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"config": self.config.to_dict()})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaPositionalEmbedding":
        cfg = HyenaDNAConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaExponentialModulation(layers.Layer):
    """Exponential window function applied to the output of the implicit filter MLP."""

    def __init__(
        self,
        config: HyenaDNAConfig,
        fast_decay_pct: float = 0.3,
        slow_decay_pct: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.05,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.d_model = int(config.d_model)
        self.shift = float(shift)
        self.fast_decay_pct = float(fast_decay_pct)
        self.slow_decay_pct = float(slow_decay_pct)
        self.target = float(target)

    def build(self, input_shape: Any = None) -> None:
        max_decay = math.log(self.target) / self.fast_decay_pct
        min_decay = math.log(self.target) / self.slow_decay_pct
        deltas_np = np.linspace(min_decay, max_decay, self.d_model, dtype="float32")[None, None, :]

        self.deltas = self.add_weight(
            name="deltas",
            shape=(1, 1, self.d_model),
            initializer=keras.initializers.Constant(deltas_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, t: keras.KerasTensor, x: keras.KerasTensor) -> keras.KerasTensor:
        decay = ops.exp(-t * ops.abs(self.deltas))
        return x * (decay + self.shift)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "config": self.config.to_dict(),
                "fast_decay_pct": self.fast_decay_pct,
                "slow_decay_pct": self.slow_decay_pct,
                "target": self.target,
                "shift": self.shift,
            }
        )
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaExponentialModulation":
        cfg = HyenaDNAConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaFilter(layers.Layer):
    """Implicit long filter predicting convolution kernels from positional encodings."""

    def __init__(
        self,
        config: HyenaDNAConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.d_model = int(config.d_model)
        self.hyena_order = int(config.hyena_order)
        self.filter_d_model = self.d_model * (self.hyena_order - 1)

        self.pos_emb = HyenaPositionalEmbedding(config, name="pos_emb")
        self.modulation = HyenaExponentialModulation(config, name="modulation")

        # Implicit filter MLP: Dense -> Sin -> [Dense -> Sin]*inner -> Dense(no bias)
        self.sin = HyenaSin(config, name="sin")
        self.dense_layers = []
        self.dense_layers.append(
            layers.Dense(config.filter_order, use_bias=True, name="dense_0")
        )
        for i in range(config.num_inner_mlps):
            self.dense_layers.append(
                layers.Dense(config.filter_order, use_bias=True, name=f"dense_{i+1}")
            )
        self.final_dense = layers.Dense(self.d_model, use_bias=False, name="final_dense")

    def build(self, input_shape: Any = None) -> None:
        self.pos_emb.build()
        self.modulation.build()
        self.sin.build((None, None, self.config.filter_order))

        cur_shape = (None, None, self.config.emb_dim)
        for layer in self.dense_layers:
            layer.build(cur_shape)
            cur_shape = (None, None, layer.units)

        self.final_dense.build(cur_shape)

        self.bias = self.add_weight(
            name="bias",
            shape=(self.filter_d_model,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def filter(self, L: Any) -> keras.KerasTensor:
        """Evaluates the implicit filter at sequence length L."""
        z, t = self.pos_emb(L)
        h = z
        for layer in self.dense_layers:
            h = self.sin(layer(h))
        h = self.final_dense(h)
        h = self.modulation(t, h)
        return h

    def call(
        self,
        x: keras.KerasTensor,
        L: Any,
        k: Optional[keras.KerasTensor] = None,
        bias: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        if k is None:
            k = self.filter(L)[0]
        if bias is None:
            bias = self.bias
        return fftconv(x, k, bias)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"config": self.config.to_dict()})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaFilter":
        cfg = HyenaDNAConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaOperator(layers.Layer):
    """Hyena recurrence operator combining short depthwise conv and implicit long conv."""

    def __init__(
        self,
        config: HyenaDNAConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.d_model = int(config.d_model)
        self.l_max = int(config.max_seq_len)
        self.order = int(config.hyena_order)
        self.inner_width = self.d_model * (self.order + 1)

        self.in_proj = layers.Dense(self.inner_width, use_bias=True, name="in_proj")
        self.short_filter = layers.DepthwiseConv1D(
            kernel_size=config.short_filter_order,
            use_bias=True,
            name="short_filter",
        )
        self.filter_fn = HyenaFilter(config, name="filter_fn")
        self.out_proj = layers.Dense(self.d_model, use_bias=True, name="out_proj")
        self.dropout = layers.Dropout(config.hyena_dropout)

    def build(self, input_shape: Any) -> None:
        self.in_proj.build(input_shape)
        proj_shape = (*input_shape[:-1], self.inner_width)
        # Depthwise 1D conv expects (batch, seq_len, channels)
        self.short_filter.build(proj_shape)
        self.filter_fn.build()
        out_shape = (*input_shape[:-1], self.d_model)
        self.out_proj.build(out_shape)
        super().build(input_shape)

    def call(
        self,
        u: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # u shape: (batch, seq_len, d_model)
        seq_len = ops.shape(u)[1]
        l_filter = ops.minimum(seq_len, self.l_max)

        # Input linear projection: (batch, seq_len, inner_width)
        u_proj = self.in_proj(u)

        # Causal short convolution: pad 2 on left of time axis
        padded = ops.pad(u_proj, [[0, 0], [2, 0], [0, 0]])
        uc = self.short_filter(padded)[:, :l_filter, :]

        # Transpose to channels-first for recurrence: (batch, inner_width, l_filter)
        uc_t = ops.transpose(uc, (0, 2, 1))
        splits = ops.split(uc_t, self.order + 1, axis=1)
        *x, v = splits

        # Generate long filter kernel and reshape: (order - 1, d_model, l_filter)
        k = self.filter_fn.filter(l_filter)[0]  # (l_filter, d_model)
        k = ops.transpose(k, (1, 0))  # (d_model, l_filter)
        k = ops.reshape(k, (self.order - 1, self.d_model, l_filter))
        bias = ops.reshape(self.filter_fn.bias, (self.order - 1, self.d_model))

        # Hyena recurrence
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i, training=training)
            v = fftconv(v, k[o], bias[o])

        # Gated output: transpose back to (batch, l_filter, d_model)
        y = ops.transpose(v * x[0], (0, 2, 1))
        return self.out_proj(y)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"config": self.config.to_dict()})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaOperator":
        cfg = HyenaDNAConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaMlp(layers.Layer):
    """Two-layer feed-forward network with tanh-approximated GELU activation."""

    def __init__(
        self,
        config: HyenaDNAConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.fc1 = layers.Dense(config.d_inner, use_bias=True, name="fc1")
        self.fc2 = layers.Dense(config.d_model, use_bias=True, name="fc2")

    def build(self, input_shape: Any) -> None:
        self.fc1.build(input_shape)
        hidden_shape = (*input_shape[:-1], self.config.d_inner)
        self.fc2.build(hidden_shape)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        y = self.fc1(x)
        # Approximate tanh GELU matching FlashAttention / Hugging Face port
        y = ops.gelu(y, approximate=True)
        return self.fc2(y)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"config": self.config.to_dict()})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaMlp":
        cfg = HyenaDNAConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)


@keras.saving.register_keras_serializable(package="keras_bio")
class HyenaBlock(layers.Layer):
    """Hyena block combining HyenaOperator and HyenaMlp with pre-layer-normalization."""

    def __init__(
        self,
        config: HyenaDNAConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.norm1 = layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="norm1"
        )
        self.mixer = HyenaOperator(config, name="mixer")
        self.norm2 = layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="norm2"
        )
        self.mlp = HyenaMlp(config, name="mlp")

    def build(self, input_shape: Any) -> None:
        self.norm1.build(input_shape)
        self.mixer.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        residual = hidden_states
        hyena_normed = self.norm1(residual)
        mixer_output = self.mixer(hyena_normed, training=training)
        residual = mixer_output + residual

        mlp_normed = self.norm2(residual)
        mlp_output = self.mlp(mlp_normed)
        return mlp_output + residual

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"config": self.config.to_dict()})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HyenaBlock":
        cfg = HyenaDNAConfig.from_dict(config.pop("config"))
        return cls(config=cfg, **config)
