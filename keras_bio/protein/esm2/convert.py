"""Utilities for converting PyTorch ESM-2 weights to Keras 3 models.

Supports loading official pretrained checkpoints from Hugging Face Hub
(e.g., `facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D`,
`facebook/esm2_t33_650M_UR50D`, etc.) or local PyTorch state dicts.
"""

import re
from typing import Any, Dict, Optional, Type, Union

import keras
from keras import ops
import numpy as np

from keras_bio.protein.esm2.config import ESM2Config


def convert_esm2_weights_from_hf(
    hf_state_dict: Dict[str, Any],
    keras_model: keras.Model,
) -> keras.Model:
    """Copies weights from a Hugging Face ESM-2 state dict into a Keras model.

    Handles transpose for 2D Dense layers (PyTorch [out_dim, in_dim] to Keras [in_dim, out_dim]),
    preserves 1D LayerNorm gamma/beta and 2D embedding tables.

    Args:
        hf_state_dict: PyTorch state dict from transformers or torch.load.
        keras_model: Instantiated and built Keras ESM-2 model.

    Returns:
        The Keras model with loaded weights.
    """
    # Normalize tensor values to numpy float32
    state_dict = {}
    for k, v in hf_state_dict.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        elif not isinstance(v, np.ndarray):
            v = np.asarray(v)
        state_dict[k] = v
        # Ensure both 'esm.' prefixed and unprefixed keys exist
        if k.startswith("esm."):
            state_dict[k[4:]] = v
        else:
            state_dict[f"esm.{k}"] = v

    for kw in keras_model.weights:
        path = kw.path
        val = None

        # 1. Embeddings
        if "word_embeddings/embeddings" in path:
            val = state_dict.get("esm.embeddings.word_embeddings.weight")
        elif "rotary_embeddings/inv_freq" in path:
            # Pre-computed buffer, keep or load
            val = state_dict.get("esm.rotary_embeddings.inv_freq", ops.convert_to_numpy(kw))

        # 2. Final LayerNorm after encoder
        elif "emb_layer_norm_after/gamma" in path:
            val = state_dict.get("esm.encoder.emb_layer_norm_after.weight")
        elif "emb_layer_norm_after/beta" in path:
            val = state_dict.get("esm.encoder.emb_layer_norm_after.bias")

        # 3. Contact Prediction Head
        elif "contact_head/regression/kernel" in path:
            w = state_dict.get("esm.contact_head.regression.weight")
            if w is not None:
                val = w.T
        elif "contact_head/regression/bias" in path:
            val = state_dict.get("esm.contact_head.regression.bias")

        # 4. Masked LM Head
        elif "lm_head/dense/kernel" in path:
            w = state_dict.get("lm_head.dense.weight")
            if w is not None:
                val = w.T
        elif "lm_head/dense/bias" in path:
            val = state_dict.get("lm_head.dense.bias")
        elif "lm_head/layer_norm/gamma" in path:
            val = state_dict.get("lm_head.layer_norm.weight")
        elif "lm_head/layer_norm/beta" in path:
            val = state_dict.get("lm_head.layer_norm.bias")
        elif "lm_head/bias" in path:
            val = state_dict.get("lm_head.bias")

        # 5. Sequence Classification Head
        elif "classifier/dense/kernel" in path:
            w = state_dict.get("classifier.dense.weight")
            if w is not None:
                val = w.T
        elif "classifier/dense/bias" in path:
            val = state_dict.get("classifier.dense.bias")
        elif "classifier/out_proj/kernel" in path:
            w = state_dict.get("classifier.out_proj.weight")
            if w is not None:
                val = w.T
        elif "classifier/out_proj/bias" in path:
            val = state_dict.get("classifier.out_proj.bias")

        # 6. Token Classification Head
        elif "classifier/kernel" in path:
            w = state_dict.get("classifier.weight")
            if w is not None:
                val = w.T
        elif "classifier/bias" in path:
            val = state_dict.get("classifier.bias")

        # 7. Per-layer encoder blocks
        else:
            match = re.search(r"layer_(\d+)/", path)
            if match:
                idx = match.group(1)
                # Self Attention
                if "attention/LayerNorm/gamma" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.attention.LayerNorm.weight")
                elif "attention/LayerNorm/beta" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.attention.LayerNorm.bias")
                elif "attention/self/query/kernel" in path:
                    w = state_dict.get(f"esm.encoder.layer.{idx}.attention.self.query.weight")
                    if w is not None:
                        val = w.T
                elif "attention/self/query/bias" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.attention.self.query.bias")
                elif "attention/self/key/kernel" in path:
                    w = state_dict.get(f"esm.encoder.layer.{idx}.attention.self.key.weight")
                    if w is not None:
                        val = w.T
                elif "attention/self/key/bias" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.attention.self.key.bias")
                elif "attention/self/value/kernel" in path:
                    w = state_dict.get(f"esm.encoder.layer.{idx}.attention.self.value.weight")
                    if w is not None:
                        val = w.T
                elif "attention/self/value/bias" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.attention.self.value.bias")
                elif "attention/output_dense/kernel" in path:
                    w = state_dict.get(f"esm.encoder.layer.{idx}.attention.output.dense.weight")
                    if w is not None:
                        val = w.T
                elif "attention/output_dense/bias" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.attention.output.dense.bias")

                # Feed Forward
                elif "feed_forward/LayerNorm/gamma" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.LayerNorm.weight")
                elif "feed_forward/LayerNorm/beta" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.LayerNorm.bias")
                elif "feed_forward/intermediate_dense/kernel" in path:
                    w = state_dict.get(f"esm.encoder.layer.{idx}.intermediate.dense.weight")
                    if w is not None:
                        val = w.T
                elif "feed_forward/intermediate_dense/bias" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.intermediate.dense.bias")
                elif "feed_forward/output_dense/kernel" in path:
                    w = state_dict.get(f"esm.encoder.layer.{idx}.output.dense.weight")
                    if w is not None:
                        val = w.T
                elif "feed_forward/output_dense/bias" in path:
                    val = state_dict.get(f"esm.encoder.layer.{idx}.output.dense.bias")

        if val is not None:
            kw.assign(val.astype(kw.dtype))

    return keras_model


def load_esm2_from_hf(
    model_cls: Type[keras.Model],
    hf_id: str = "facebook/esm2_t6_8M_UR50D",
    dtype: str = "float32",
    **kwargs: Any,
) -> keras.Model:
    """Loads a pretrained ESM-2 model directly from Hugging Face Hub.

    Args:
        model_cls: Keras model class (ESM2Model, ESM2ForMaskedLM, etc.).
        hf_id: Hugging Face repository ID (e.g. "facebook/esm2_t6_8M_UR50D").
        dtype: Target weight precision ("float32", "float16", "bfloat16").
        kwargs: Additional arguments passed to model_cls.

    Returns:
        Loaded Keras model with pretrained weights.
    """
    import transformers

    hf_config = transformers.AutoConfig.from_pretrained(hf_id)
    cfg = ESM2Config.from_hf(hf_config.to_dict())

    # Build model
    model = model_cls(config=cfg, dtype=dtype, **kwargs)
    model.build((None, 12))

    # Download and load PyTorch state dict
    try:
        from transformers import EsmForMaskedLM

        hf_pt_model = EsmForMaskedLM.from_pretrained(hf_id)
        state_dict = hf_pt_model.state_dict()
    except Exception:
        # Fallback to safetensors or torch download
        from huggingface_hub import hf_hub_download
        import torch

        try:
            weights_path = hf_hub_download(repo_id=hf_id, filename="model.safetensors")
            from safetensors.torch import load_file

            state_dict = load_file(weights_path)
        except Exception:
            weights_path = hf_hub_download(repo_id=hf_id, filename="pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")

    convert_esm2_weights_from_hf(state_dict, model)
    return model
