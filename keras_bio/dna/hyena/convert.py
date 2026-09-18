"""Utilities for converting PyTorch HyenaDNA weights to Keras 3 models.

Supports loading official pretrained checkpoints from Hugging Face Hub
(e.g., `LongSafari/hyenadna-tiny-1k-seqlen-hf`, `LongSafari/hyenadna-small-32k-seqlen-hf`,
`LongSafari/hyenadna-medium-160k-seqlen-hf`, `LongSafari/hyenadna-large-1m-seqlen-hf`)
or local PyTorch state dicts.
"""

import re
from typing import Any, Dict, Optional, Type, Union

import keras
from keras import ops
import numpy as np

from keras_bio.dna.hyena.config import HyenaDNAConfig


def convert_hyenadna_weights_from_hf(
    hf_state_dict: Dict[str, Any],
    keras_model: keras.Model,
) -> keras.Model:
    """Copies weights from a Hugging Face HyenaDNA state dict into a Keras model.

    Handles transposing 2D Dense layers from PyTorch [out_features, in_features]
    to Keras [in_features, out_features], converting depthwise 1D conv filters
    from [channels, 1, kernel_size] to [kernel_size, channels, 1], and copying
    buffers (positional embeddings and modulation deltas).

    Args:
        hf_state_dict: PyTorch state dict from transformers or safetensors.
        keras_model: Instantiated and built Keras HyenaDNA model.

    Returns:
        The Keras model with loaded weights.
    """
    state_dict = {}
    for k, v in hf_state_dict.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        elif not isinstance(v, np.ndarray):
            v = np.asarray(v)
        state_dict[k] = v
        # Support prefixed ("hyena.") and unprefixed keys
        if k.startswith("hyena."):
            state_dict[k[6:]] = v
        else:
            state_dict[f"hyena.{k}"] = v

    for kw in keras_model.weights:
        path = kw.path
        val = None

        # 1. Word Embeddings
        if "word_embeddings/embeddings" in path:
            val = state_dict.get("hyena.backbone.embeddings.word_embeddings.weight")

        # 2. Final LayerNorm
        elif "ln_f/gamma" in path:
            val = state_dict.get("hyena.backbone.ln_f.weight")
        elif "ln_f/beta" in path:
            val = state_dict.get("hyena.backbone.ln_f.bias")

        # 3. LM Head
        elif "lm_head/kernel" in path:
            w = state_dict.get("lm_head.weight")
            if w is not None:
                val = w.T

        # 4. Sequence Classifier
        elif "score/kernel" in path:
            w = state_dict.get("score.weight")
            if w is not None:
                val = w.T

        # 5. Hyena Blocks
        else:
            match = re.search(r"layer_(\d+)/", path)
            if match:
                idx = match.group(1)
                prefix = f"hyena.backbone.layers.{idx}"

                # LayerNorms
                if "norm1/gamma" in path:
                    val = state_dict.get(f"{prefix}.norm1.weight")
                elif "norm1/beta" in path:
                    val = state_dict.get(f"{prefix}.norm1.bias")
                elif "norm2/gamma" in path:
                    val = state_dict.get(f"{prefix}.norm2.weight")
                elif "norm2/beta" in path:
                    val = state_dict.get(f"{prefix}.norm2.bias")

                # MLP
                elif "mlp/fc1/kernel" in path:
                    w = state_dict.get(f"{prefix}.mlp.fc1.weight")
                    if w is not None:
                        val = w.T
                elif "mlp/fc1/bias" in path:
                    val = state_dict.get(f"{prefix}.mlp.fc1.bias")
                elif "mlp/fc2/kernel" in path:
                    w = state_dict.get(f"{prefix}.mlp.fc2.weight")
                    if w is not None:
                        val = w.T
                elif "mlp/fc2/bias" in path:
                    val = state_dict.get(f"{prefix}.mlp.fc2.bias")

                # Mixer (HyenaOperator)
                elif "mixer/in_proj/kernel" in path:
                    w = state_dict.get(f"{prefix}.mixer.in_proj.weight")
                    if w is not None:
                        val = w.T
                elif "mixer/in_proj/bias" in path:
                    val = state_dict.get(f"{prefix}.mixer.in_proj.bias")
                elif "mixer/out_proj/kernel" in path:
                    w = state_dict.get(f"{prefix}.mixer.out_proj.weight")
                    if w is not None:
                        val = w.T
                elif "mixer/out_proj/bias" in path:
                    val = state_dict.get(f"{prefix}.mixer.out_proj.bias")

                # Depthwise short filter conv1d: (channels, 1, 3) -> (3, channels, 1)
                elif "mixer/short_filter/kernel" in path:
                    w = state_dict.get(f"{prefix}.mixer.short_filter.weight")
                    if w is not None:
                        val = np.transpose(w, (2, 0, 1))
                elif "mixer/short_filter/bias" in path:
                    val = state_dict.get(f"{prefix}.mixer.short_filter.bias")

                # Implicit filter & buffers
                elif "mixer/filter_fn/bias" in path:
                    val = state_dict.get(f"{prefix}.mixer.filter_fn.bias")
                elif "mixer/filter_fn/pos_emb/z" in path:
                    val = state_dict.get(f"{prefix}.mixer.filter_fn.pos_emb.z")
                elif "mixer/filter_fn/pos_emb/t" in path:
                    val = state_dict.get(f"{prefix}.mixer.filter_fn.pos_emb.t")
                elif "mixer/filter_fn/modulation/deltas" in path:
                    val = state_dict.get(f"{prefix}.mixer.filter_fn.modulation.deltas")
                elif "mixer/filter_fn/sin/freq" in path:
                    val = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.1.freq"
                    )

                # Dense layers inside implicit filter MLP
                elif "mixer/filter_fn/dense_0/kernel" in path:
                    w = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.0.weight"
                    )
                    if w is not None:
                        val = w.T
                elif "mixer/filter_fn/dense_0/bias" in path:
                    val = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.0.bias"
                    )
                elif "mixer/filter_fn/dense_1/kernel" in path:
                    w = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.2.weight"
                    )
                    if w is not None:
                        val = w.T
                elif "mixer/filter_fn/dense_1/bias" in path:
                    val = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.2.bias"
                    )
                elif "mixer/filter_fn/dense_2/kernel" in path:
                    w = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.4.weight"
                    )
                    if w is not None:
                        val = w.T
                elif "mixer/filter_fn/dense_2/bias" in path:
                    val = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.4.bias"
                    )
                elif "mixer/filter_fn/final_dense/kernel" in path:
                    w = state_dict.get(
                        f"{prefix}.mixer.filter_fn.implicit_filter.6.weight"
                    )
                    if w is not None:
                        val = w.T

        if val is not None:
            kw.assign(val.astype(kw.dtype))

    return keras_model


def load_hyenadna_from_hf(
    model_cls: Type[keras.Model],
    hf_id: str = "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    dtype: str = "float32",
    **kwargs: Any,
) -> keras.Model:
    """Loads a pretrained HyenaDNA model directly from Hugging Face Hub.

    Args:
        model_cls: Keras model class (HyenaDNAModel, HyenaDNAForCausalLM, etc.).
        hf_id: Hugging Face repository ID (e.g. "LongSafari/hyenadna-tiny-1k-seqlen-hf").
        dtype: Target weight precision ("float32", "float16", "bfloat16").
        kwargs: Additional arguments passed to model_cls.

    Returns:
        Loaded Keras model with pretrained weights.
    """
    clean_name = hf_id.replace("LongSafari/", "")
    try:
        import json
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id=hf_id, filename="config.json")
        with open(cfg_path) as f:
            raw_cfg = json.load(f)
        cfg = HyenaDNAConfig.from_hf(raw_cfg)
    except Exception:
        cfg = HyenaDNAConfig.from_preset(clean_name)

    # Build model
    model = model_cls(config=cfg, dtype=dtype, **kwargs)
    model.build((None, 16))

    # Download safetensors or bin
    from huggingface_hub import hf_hub_download

    try:
        weights_path = hf_hub_download(repo_id=hf_id, filename="model.safetensors")
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
    except Exception:
        import torch

        weights_path = hf_hub_download(repo_id=hf_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")

    convert_hyenadna_weights_from_hf(state_dict, model)
    return model

