"""Utilities for converting PyTorch DNABERT weights to Keras 3 models.

Supports loading official pretrained checkpoints from Hugging Face Hub
(e.g., `zhihan1996/DNA_bert_3`, `zhihan1996/DNA_bert_4`, `zhihan1996/DNA_bert_5`,
`zhihan1996/DNA_bert_6`) or local PyTorch state dicts.
"""

import re
from typing import Any, Dict, Optional, Type, Union

import keras
from keras import ops
import numpy as np

from keras_bio.dna.dnabert.config import DNABERTConfig


def convert_dnabert_weights_from_hf(
    hf_state_dict: Dict[str, Any],
    keras_model: keras.Model,
) -> keras.Model:
    """Copies weights from a Hugging Face DNABERT state dict into a Keras model.

    Handles transposing 2D Dense layers from PyTorch [out_features, in_features]
    to Keras [in_features, out_features], preserves 1D LayerNorm gamma/beta,
    and 2D embedding tables.

    Args:
        hf_state_dict: PyTorch state dict from transformers or torch.load.
        keras_model: Instantiated and built Keras DNABERT model.

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
        # Ensure both 'bert.' prefixed and unprefixed keys exist
        if k.startswith("bert."):
            state_dict[k[5:]] = v
        else:
            state_dict[f"bert.{k}"] = v

    for kw in keras_model.weights:
        path = kw.path
        val = None

        # 1. Embeddings
        if "word_embeddings/embeddings" in path:
            val = state_dict.get("embeddings.word_embeddings.weight")
        elif "position_embeddings/embeddings" in path:
            val = state_dict.get("embeddings.position_embeddings.weight")
        elif "token_type_embeddings/embeddings" in path:
            val = state_dict.get("embeddings.token_type_embeddings.weight")
        elif "embeddings/LayerNorm/gamma" in path:
            val = state_dict.get("embeddings.LayerNorm.weight")
        elif "embeddings/LayerNorm/beta" in path:
            val = state_dict.get("embeddings.LayerNorm.bias")

        # 2. Pooler
        elif "pooler/dense/kernel" in path:
            w = state_dict.get("pooler.dense.weight")
            if w is not None:
                val = w.T
        elif "pooler/dense/bias" in path:
            val = state_dict.get("pooler.dense.bias")

        # 3. Masked LM Head
        elif "lm_head/transform/dense/kernel" in path:
            w = state_dict.get("cls.predictions.transform.dense.weight")
            if w is not None:
                val = w.T
        elif "lm_head/transform/dense/bias" in path:
            val = state_dict.get("cls.predictions.transform.dense.bias")
        elif "lm_head/transform/LayerNorm/gamma" in path:
            val = state_dict.get("cls.predictions.transform.LayerNorm.weight")
        elif "lm_head/transform/LayerNorm/beta" in path:
            val = state_dict.get("cls.predictions.transform.LayerNorm.bias")
        elif "lm_head/bias" in path:
            val = state_dict.get("cls.predictions.bias")
            if val is None:
                val = state_dict.get("cls.predictions.decoder.bias")

        # 4. Classification Heads
        elif "classifier/kernel" in path:
            w = state_dict.get("classifier.weight")
            if w is not None:
                val = w.T
        elif "classifier/bias" in path:
            val = state_dict.get("classifier.bias")

        # 5. Transformer Encoder Layers
        else:
            match = re.search(r"layer_(\d+)/", path)
            if match:
                idx = match.group(1)
                # Self Attention
                if "attention/self/query/kernel" in path:
                    w = state_dict.get(f"encoder.layer.{idx}.attention.self.query.weight")
                    if w is not None:
                        val = w.T
                elif "attention/self/query/bias" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.attention.self.query.bias")
                elif "attention/self/key/kernel" in path:
                    w = state_dict.get(f"encoder.layer.{idx}.attention.self.key.weight")
                    if w is not None:
                        val = w.T
                elif "attention/self/key/bias" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.attention.self.key.bias")
                elif "attention/self/value/kernel" in path:
                    w = state_dict.get(f"encoder.layer.{idx}.attention.self.value.weight")
                    if w is not None:
                        val = w.T
                elif "attention/self/value/bias" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.attention.self.value.bias")
                elif "attention/dense/kernel" in path:
                    w = state_dict.get(f"encoder.layer.{idx}.attention.output.dense.weight")
                    if w is not None:
                        val = w.T
                elif "attention/dense/bias" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.attention.output.dense.bias")
                elif "attention/LayerNorm/gamma" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.attention.output.LayerNorm.weight")
                elif "attention/LayerNorm/beta" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.attention.output.LayerNorm.bias")

                # Feed-Forward Intermediate & Output
                elif "intermediate/dense/kernel" in path:
                    w = state_dict.get(f"encoder.layer.{idx}.intermediate.dense.weight")
                    if w is not None:
                        val = w.T
                elif "intermediate/dense/bias" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.intermediate.dense.bias")
                elif "output/dense/kernel" in path:
                    w = state_dict.get(f"encoder.layer.{idx}.output.dense.weight")
                    if w is not None:
                        val = w.T
                elif "output/dense/bias" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.output.dense.bias")
                elif "output/LayerNorm/gamma" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.output.LayerNorm.weight")
                elif "output/LayerNorm/beta" in path:
                    val = state_dict.get(f"encoder.layer.{idx}.output.LayerNorm.bias")

        if val is not None:
            kw.assign(val.astype(kw.dtype))

    return keras_model


def load_dnabert_from_hf(
    model_cls: Type[keras.Model],
    hf_id: str = "zhihan1996/DNA_bert_6",
    dtype: str = "float32",
    **kwargs: Any,
) -> keras.Model:
    """Loads a pretrained DNABERT model directly from Hugging Face Hub.

    Args:
        model_cls: Keras model class (DNABERTModel, DNABERTForMaskedLM, etc.).
        hf_id: Hugging Face repository ID (e.g. "zhihan1996/DNA_bert_6").
        dtype: Target weight precision ("float32", "float16", "bfloat16").
        kwargs: Additional arguments passed to model_cls.

    Returns:
        Loaded Keras model with pretrained weights.
    """
    clean_name = hf_id.replace("zhihan1996/", "")
    try:
        import transformers

        hf_config = transformers.AutoConfig.from_pretrained(hf_id)
        cfg = DNABERTConfig.from_hf(hf_config.to_dict())
    except Exception:
        # Fallback to local preset if offline or transformers error
        cfg = DNABERTConfig.from_preset(clean_name)

    # Build model
    model = model_cls(config=cfg, dtype=dtype, **kwargs)
    model.build((None, 16))

    # Download and load PyTorch state dict
    try:
        from transformers import BertForMaskedLM

        hf_pt_model = BertForMaskedLM.from_pretrained(hf_id)
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

    convert_dnabert_weights_from_hf(state_dict, model)
    return model

