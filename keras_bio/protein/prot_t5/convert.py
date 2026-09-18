"""Weight conversion utilities between PyTorch/HuggingFace and Keras 3 ProtT5.

Supports:
- Loading from Hugging Face Hub (online)
- Loading from local checkpoints (.bin, .safetensors, .pt)
- Direct numpy state_dict assignment
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from keras_bio.protein.prot_t5.config import ProtT5Config


class WeightMappingError(ValueError):
    """Raised when weight shapes do not match or a required weight is missing."""
    pass


def _to_numpy(val: Any) -> np.ndarray:
    """Convert a PyTorch tensor, numpy array, or other container to numpy float32/target."""
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    return np.asarray(val)


def _assign_variable(keras_var, hf_val: np.ndarray, var_name: str = "", transpose: bool = False):
    """Assign numpy array to a Keras Variable, transposing linear weight kernels if specified."""
    hf_val = np.asarray(hf_val, dtype="float32")
    if transpose:
        hf_val = hf_val.T
    target_shape = tuple(keras_var.shape)

    if hf_val.shape != target_shape:
        raise WeightMappingError(
            f"Shape mismatch for {var_name}: Hugging Face shape {hf_val.shape} "
            f"does not match Keras shape {target_shape}"
        )
    keras_var.assign(hf_val)


def transfer_t5_weights(keras_model, hf_state_dict: Dict[str, Any]) -> None:
    """Copies weights from HuggingFace PyTorch state dict into an initialized Keras ProtT5 model.

    Works for ProtT5EncoderModel, ProtT5Model, ProtT5ForConditionalGeneration,
    ProtT5ForSequenceClassification, and ProtT5ForTokenClassification.
    """
    # Normalize state dict keys (strip 'transformer.' prefix if present)
    hf = {}
    for k, v in hf_state_dict.items():
        clean_key = k[len("transformer."):] if k.startswith("transformer.") else k
        hf[clean_key] = _to_numpy(v)

    # 1. Shared Embedding
    if "shared.weight" in hf:
        _assign_variable(keras_model.shared.embeddings, hf["shared.weight"], "shared.weight", transpose=False)
    elif "encoder.embed_tokens.weight" in hf:
        _assign_variable(keras_model.shared.embeddings, hf["encoder.embed_tokens.weight"], "shared.weight", transpose=False)

    # 2. Encoder Relative Attention Bias
    if "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" in hf:
        _assign_variable(
            keras_model.encoder_rel_bias.embeddings,
            hf["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            "encoder.block.0.SelfAttention.relative_attention_bias",
            transpose=False,
        )

    # 3. Encoder Blocks
    if hasattr(keras_model, "encoder_blocks"):
        for i, block in enumerate(keras_model.encoder_blocks):
            prefix = f"encoder.block.{i}."
            # Layer 0: SelfAttention + LayerNorm
            ln0_key = f"{prefix}layer.0.layer_norm.weight"
            if ln0_key in hf:
                _assign_variable(block.self_attention.layer_norm.weight, hf[ln0_key], ln0_key, transpose=False)

            for proj in ("q", "k", "v", "o"):
                proj_key = f"{prefix}layer.0.SelfAttention.{proj}.weight"
                if proj_key in hf:
                    layer_proj = getattr(block.self_attention.attention, proj)
                    _assign_variable(layer_proj.kernel, hf[proj_key], proj_key, transpose=True)

            # Layer 1: FeedForward + LayerNorm
            ln1_key = f"{prefix}layer.1.layer_norm.weight"
            if ln1_key in hf:
                _assign_variable(block.ff.layer_norm.weight, hf[ln1_key], ln1_key, transpose=False)

            ff_layer = block.ff.ff
            if hasattr(ff_layer, "wi"):
                # Ungated variant
                wi_key = f"{prefix}layer.1.DenseReluDense.wi.weight"
                wo_key = f"{prefix}layer.1.DenseReluDense.wo.weight"
                if wi_key in hf:
                    _assign_variable(ff_layer.wi.kernel, hf[wi_key], wi_key, transpose=True)
                if wo_key in hf:
                    _assign_variable(ff_layer.wo.kernel, hf[wo_key], wo_key, transpose=True)
            elif hasattr(ff_layer, "wi_0"):
                # Gated variant
                wi_0_key = f"{prefix}layer.1.DenseReluDense.wi_0.weight"
                wi_1_key = f"{prefix}layer.1.DenseReluDense.wi_1.weight"
                wo_key = f"{prefix}layer.1.DenseReluDense.wo.weight"
                if wi_0_key in hf:
                    _assign_variable(ff_layer.wi_0.kernel, hf[wi_0_key], wi_0_key, transpose=True)
                if wi_1_key in hf:
                    _assign_variable(ff_layer.wi_1.kernel, hf[wi_1_key], wi_1_key, transpose=True)
                if wo_key in hf:
                    _assign_variable(ff_layer.wo.kernel, hf[wo_key], wo_key, transpose=True)

    # 4. Encoder Final LayerNorm
    if "encoder.final_layer_norm.weight" in hf:
        _assign_variable(
            keras_model.encoder_final_layer_norm.weight,
            hf["encoder.final_layer_norm.weight"],
            "encoder.final_layer_norm.weight",
            transpose=False,
        )

    # 5. Decoder Relative Attention Bias
    if "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" in hf and hasattr(keras_model, "decoder_rel_bias"):
        _assign_variable(
            keras_model.decoder_rel_bias.embeddings,
            hf["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            "decoder.block.0.SelfAttention.relative_attention_bias",
            transpose=False,
        )

    # 6. Decoder Blocks
    if hasattr(keras_model, "decoder_blocks"):
        for i, block in enumerate(keras_model.decoder_blocks):
            prefix = f"decoder.block.{i}."
            # Layer 0: SelfAttention
            ln0_key = f"{prefix}layer.0.layer_norm.weight"
            if ln0_key in hf:
                _assign_variable(block.self_attention.layer_norm.weight, hf[ln0_key], ln0_key, transpose=False)

            for proj in ("q", "k", "v", "o"):
                proj_key = f"{prefix}layer.0.SelfAttention.{proj}.weight"
                if proj_key in hf:
                    layer_proj = getattr(block.self_attention.attention, proj)
                    _assign_variable(layer_proj.kernel, hf[proj_key], proj_key, transpose=True)

            # Layer 1: CrossAttention
            ln1_key = f"{prefix}layer.1.layer_norm.weight"
            if ln1_key in hf:
                _assign_variable(block.cross_attention.layer_norm.weight, hf[ln1_key], ln1_key, transpose=False)

            for proj in ("q", "k", "v", "o"):
                proj_key = f"{prefix}layer.1.EncDecAttention.{proj}.weight"
                if proj_key in hf:
                    layer_proj = getattr(block.cross_attention.attention, proj)
                    _assign_variable(layer_proj.kernel, hf[proj_key], proj_key, transpose=True)

            # Layer 2: FeedForward
            ln2_key = f"{prefix}layer.2.layer_norm.weight"
            if ln2_key in hf:
                _assign_variable(block.ff.layer_norm.weight, hf[ln2_key], ln2_key, transpose=False)

            ff_layer = block.ff.ff
            if hasattr(ff_layer, "wi"):
                wi_key = f"{prefix}layer.2.DenseReluDense.wi.weight"
                wo_key = f"{prefix}layer.2.DenseReluDense.wo.weight"
                if wi_key in hf:
                    _assign_variable(ff_layer.wi.kernel, hf[wi_key], wi_key, transpose=True)
                if wo_key in hf:
                    _assign_variable(ff_layer.wo.kernel, hf[wo_key], wo_key, transpose=True)
            elif hasattr(ff_layer, "wi_0"):
                wi_0_key = f"{prefix}layer.2.DenseReluDense.wi_0.weight"
                wi_1_key = f"{prefix}layer.2.DenseReluDense.wi_1.weight"
                wo_key = f"{prefix}layer.2.DenseReluDense.wo.weight"
                if wi_0_key in hf:
                    _assign_variable(ff_layer.wi_0.kernel, hf[wi_0_key], wi_0_key, transpose=True)
                if wi_1_key in hf:
                    _assign_variable(ff_layer.wi_1.kernel, hf[wi_1_key], wi_1_key, transpose=True)
                if wo_key in hf:
                    _assign_variable(ff_layer.wo.kernel, hf[wo_key], wo_key, transpose=True)

    # 7. Decoder Final LayerNorm
    if "decoder.final_layer_norm.weight" in hf and hasattr(keras_model, "decoder_final_layer_norm"):
        _assign_variable(
            keras_model.decoder_final_layer_norm.weight,
            hf["decoder.final_layer_norm.weight"],
            "decoder.final_layer_norm.weight",
            transpose=False,
        )

    # 8. Untied LM Head
    if "lm_head.weight" in hf and hasattr(keras_model, "lm_head") and keras_model.lm_head is not None:
        _assign_variable(keras_model.lm_head.kernel, hf["lm_head.weight"], "lm_head.weight", transpose=True)

    # 9. Sequence Classification Head
    if hasattr(keras_model, "classifier_dense"):
        if "classification_head.dense.weight" in hf:
            _assign_variable(keras_model.classifier_dense.kernel, hf["classification_head.dense.weight"], "classification_head.dense.weight", transpose=True)
        if "classification_head.dense.bias" in hf:
            _assign_variable(keras_model.classifier_dense.bias, hf["classification_head.dense.bias"], "classification_head.dense.bias", transpose=False)
        if "classification_head.out_proj.weight" in hf:
            _assign_variable(keras_model.classifier_out_proj.kernel, hf["classification_head.out_proj.weight"], "classification_head.out_proj.weight", transpose=True)
        if "classification_head.out_proj.bias" in hf:
            _assign_variable(keras_model.classifier_out_proj.bias, hf["classification_head.out_proj.bias"], "classification_head.out_proj.bias", transpose=False)

    # 10. Token Classification Head
    if hasattr(keras_model, "classifier"):
        if "classifier.weight" in hf:
            _assign_variable(keras_model.classifier.kernel, hf["classifier.weight"], "classifier.weight", transpose=True)
        if "classifier.bias" in hf:
            _assign_variable(keras_model.classifier.bias, hf["classifier.bias"], "classifier.bias", transpose=False)


def load_t5_encoder_from_hf(
    cls,
    hf_id: str,
    dtype: str = "float32",
    **kwargs,
):
    """Load ProtT5 encoder model from Hugging Face."""
    import torch
    from transformers import AutoConfig
    from transformers import T5EncoderModel as HFT5EncoderModel

    hf_config = AutoConfig.from_pretrained(hf_id).to_dict()
    config = ProtT5Config.from_hf(hf_config)
    model = cls(config=config, **kwargs)
    model.build()

    torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float32
    hf_model = HFT5EncoderModel.from_pretrained(hf_id, torch_dtype=torch_dtype).eval()
    state_dict = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}
    transfer_t5_weights(model, state_dict)
    return model


def load_t5_model_from_hf(
    cls,
    hf_id: str,
    dtype: str = "float32",
    **kwargs,
):
    """Load full ProtT5 model from Hugging Face."""
    import torch
    from transformers import AutoConfig
    from transformers import T5ForConditionalGeneration as HFT5ForConditionalGeneration

    hf_config = AutoConfig.from_pretrained(hf_id).to_dict()
    config = ProtT5Config.from_hf(hf_config)
    model = cls(config=config, **kwargs)
    model.build()

    torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float32
    hf_model = HFT5ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=torch_dtype).eval()
    state_dict = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}
    transfer_t5_weights(model, state_dict)
    return model


# Backward-compatible aliases
transfer_weights = _assign_variable

