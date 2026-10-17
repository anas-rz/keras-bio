"""
Weight conversion utilities for porting Hugging Face PyTorch ESMFold models to Keras 3.
"""

from typing import Any, Optional, Union
import numpy as np

import keras
from keras import ops

from keras_bio.protein.esm2.convert import convert_esm2_weights_from_hf
from keras_bio.structure.esmfold.config import ESMFoldConfig
from keras_bio.structure.esmfold.modeling import ESMFoldModel


def _to_numpy(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "detach"):
        return tensor_or_array.detach().cpu().numpy()
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    return np.asarray(tensor_or_array)


def convert_esmfold_weights_from_hf(
    hf_state_dict: dict[str, Any],
    keras_model: ESMFoldModel,
) -> ESMFoldModel:
    """
    Copies weights from a Hugging Face ESMFold (EsmForProteinFolding) PyTorch state dict into a Keras ESMFoldModel.

    Args:
        hf_state_dict: PyTorch state dict dictionary.
        keras_model: Instantiated and built ESMFoldModel.

    Returns:
        The Keras model with loaded weights.
    """
    # 1. Convert ESM-2 language model backbone weights
    convert_esm2_weights_from_hf(hf_state_dict, keras_model.esm)

    # 2. ESM layer combination parameter
    if "esm_s_combine" in hf_state_dict:
        keras_model.esm_s_combine.assign(_to_numpy(hf_state_dict["esm_s_combine"]))

    # 3. esm_s_mlp
    if "esm_s_mlp.0.weight" in hf_state_dict:
        keras_model.esm_s_mlp_norm.gamma.assign(_to_numpy(hf_state_dict["esm_s_mlp.0.weight"]))
        keras_model.esm_s_mlp_norm.beta.assign(_to_numpy(hf_state_dict["esm_s_mlp.0.bias"]))
    if "esm_s_mlp.1.weight" in hf_state_dict:
        keras_model.esm_s_mlp_dense1.kernel.assign(_to_numpy(hf_state_dict["esm_s_mlp.1.weight"]).T)
        keras_model.esm_s_mlp_dense1.bias.assign(_to_numpy(hf_state_dict["esm_s_mlp.1.bias"]))
    if "esm_s_mlp.3.weight" in hf_state_dict:
        keras_model.esm_s_mlp_dense2.kernel.assign(_to_numpy(hf_state_dict["esm_s_mlp.3.weight"]).T)
        keras_model.esm_s_mlp_dense2.bias.assign(_to_numpy(hf_state_dict["esm_s_mlp.3.bias"]))

    # 4. Optional amino acid embedding
    if keras_model.embedding is not None and "embedding.weight" in hf_state_dict:
        keras_model.embedding.embeddings.assign(_to_numpy(hf_state_dict["embedding.weight"]))

    # 5. Trunk recycling & projections
    trunk = keras_model.trunk
    if "trunk.pairwise_positional_embedding.embedding.weight" in hf_state_dict:
        trunk.pairwise_positional_embedding.embedding.embeddings.assign(
            _to_numpy(hf_state_dict["trunk.pairwise_positional_embedding.embedding.weight"])
        )
    if "trunk.recycle_s_norm.weight" in hf_state_dict:
        trunk.recycle_s_norm.gamma.assign(_to_numpy(hf_state_dict["trunk.recycle_s_norm.weight"]))
        trunk.recycle_s_norm.beta.assign(_to_numpy(hf_state_dict["trunk.recycle_s_norm.bias"]))
    if "trunk.recycle_z_norm.weight" in hf_state_dict:
        trunk.recycle_z_norm.gamma.assign(_to_numpy(hf_state_dict["trunk.recycle_z_norm.weight"]))
        trunk.recycle_z_norm.beta.assign(_to_numpy(hf_state_dict["trunk.recycle_z_norm.bias"]))
    if "trunk.recycle_disto.weight" in hf_state_dict:
        trunk.recycle_disto.embeddings.assign(_to_numpy(hf_state_dict["trunk.recycle_disto.weight"]))
    if "trunk.trunk2sm_s.weight" in hf_state_dict:
        trunk.trunk2sm_s.kernel.assign(_to_numpy(hf_state_dict["trunk.trunk2sm_s.weight"]).T)
        trunk.trunk2sm_s.bias.assign(_to_numpy(hf_state_dict["trunk.trunk2sm_s.bias"]))
    if "trunk.trunk2sm_z.weight" in hf_state_dict:
        trunk.trunk2sm_z.kernel.assign(_to_numpy(hf_state_dict["trunk.trunk2sm_z.weight"]).T)
        trunk.trunk2sm_z.bias.assign(_to_numpy(hf_state_dict["trunk.trunk2sm_z.bias"]))

    # 6. Trunk blocks
    for i, block in enumerate(trunk.blocks):
        prefix = f"trunk.blocks.{i}."
        # pair_to_sequence
        if f"{prefix}pair_to_sequence.layernorm.weight" in hf_state_dict:
            block.pair_to_sequence.layernorm.gamma.assign(_to_numpy(hf_state_dict[f"{prefix}pair_to_sequence.layernorm.weight"]))
            block.pair_to_sequence.layernorm.beta.assign(_to_numpy(hf_state_dict[f"{prefix}pair_to_sequence.layernorm.bias"]))
        if f"{prefix}pair_to_sequence.linear.weight" in hf_state_dict:
            block.pair_to_sequence.linear.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}pair_to_sequence.linear.weight"]).T)

        # layernorm_1
        if f"{prefix}layernorm_1.weight" in hf_state_dict:
            block.layernorm_1.gamma.assign(_to_numpy(hf_state_dict[f"{prefix}layernorm_1.weight"]))
            block.layernorm_1.beta.assign(_to_numpy(hf_state_dict[f"{prefix}layernorm_1.bias"]))

        # seq_attention
        if f"{prefix}seq_attention.proj.weight" in hf_state_dict:
            block.seq_attention.proj.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}seq_attention.proj.weight"]).T)
        if f"{prefix}seq_attention.o_proj.weight" in hf_state_dict:
            block.seq_attention.o_proj.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}seq_attention.o_proj.weight"]).T)
            block.seq_attention.o_proj.bias.assign(_to_numpy(hf_state_dict[f"{prefix}seq_attention.o_proj.bias"]))
        if f"{prefix}seq_attention.g_proj.weight" in hf_state_dict and block.seq_attention.g_proj is not None:
            block.seq_attention.g_proj.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}seq_attention.g_proj.weight"]).T)
            block.seq_attention.g_proj.bias.assign(_to_numpy(hf_state_dict[f"{prefix}seq_attention.g_proj.bias"]))

        # mlp_seq
        if f"{prefix}mlp_seq.mlp.0.weight" in hf_state_dict:
            block.mlp_seq.layernorm.gamma.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_seq.mlp.0.weight"]))
            block.mlp_seq.layernorm.beta.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_seq.mlp.0.bias"]))
            block.mlp_seq.linear_1.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_seq.mlp.1.weight"]).T)
            block.mlp_seq.linear_1.bias.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_seq.mlp.1.bias"]))
            block.mlp_seq.linear_2.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_seq.mlp.3.weight"]).T)
            block.mlp_seq.linear_2.bias.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_seq.mlp.3.bias"]))

        # sequence_to_pair
        if f"{prefix}sequence_to_pair.layernorm.weight" in hf_state_dict:
            block.sequence_to_pair.layernorm.gamma.assign(_to_numpy(hf_state_dict[f"{prefix}sequence_to_pair.layernorm.weight"]))
            block.sequence_to_pair.layernorm.beta.assign(_to_numpy(hf_state_dict[f"{prefix}sequence_to_pair.layernorm.bias"]))
            block.sequence_to_pair.proj.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}sequence_to_pair.proj.weight"]).T)
            block.sequence_to_pair.proj.bias.assign(_to_numpy(hf_state_dict[f"{prefix}sequence_to_pair.proj.bias"]))
            block.sequence_to_pair.o_proj.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}sequence_to_pair.o_proj.weight"]).T)
            block.sequence_to_pair.o_proj.bias.assign(_to_numpy(hf_state_dict[f"{prefix}sequence_to_pair.o_proj.bias"]))

        # tri_mul_out & tri_mul_in
        for tri_name, tri_layer in [("tri_mul_out", block.tri_mul_out), ("tri_mul_in", block.tri_mul_in)]:
            t_pref = f"{prefix}{tri_name}."
            if f"{t_pref}layer_norm_in.weight" in hf_state_dict:
                tri_layer.layer_norm_in.gamma.assign(_to_numpy(hf_state_dict[f"{t_pref}layer_norm_in.weight"]))
                tri_layer.layer_norm_in.beta.assign(_to_numpy(hf_state_dict[f"{t_pref}layer_norm_in.bias"]))
                tri_layer.layer_norm_out.gamma.assign(_to_numpy(hf_state_dict[f"{t_pref}layer_norm_out.weight"]))
                tri_layer.layer_norm_out.beta.assign(_to_numpy(hf_state_dict[f"{t_pref}layer_norm_out.bias"]))

                for d_name in ["linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g", "linear_g", "linear_z"]:
                    d_layer = getattr(tri_layer, d_name)
                    d_layer.kernel.assign(_to_numpy(hf_state_dict[f"{t_pref}{d_name}.weight"]).T)
                    d_layer.bias.assign(_to_numpy(hf_state_dict[f"{t_pref}{d_name}.bias"]))

        # tri_att_start & tri_att_end
        for att_name, att_layer in [("tri_att_start", block.tri_att_start), ("tri_att_end", block.tri_att_end)]:
            a_pref = f"{prefix}{att_name}."
            if f"{a_pref}layer_norm.weight" in hf_state_dict:
                att_layer.layer_norm.gamma.assign(_to_numpy(hf_state_dict[f"{a_pref}layer_norm.weight"]))
                att_layer.layer_norm.beta.assign(_to_numpy(hf_state_dict[f"{a_pref}layer_norm.bias"]))
                att_layer.linear.kernel.assign(_to_numpy(hf_state_dict[f"{a_pref}linear.weight"]).T)

                mha = att_layer.mha
                mha.linear_q.kernel.assign(_to_numpy(hf_state_dict[f"{a_pref}mha.linear_q.weight"]).T)
                mha.linear_k.kernel.assign(_to_numpy(hf_state_dict[f"{a_pref}mha.linear_k.weight"]).T)
                mha.linear_v.kernel.assign(_to_numpy(hf_state_dict[f"{a_pref}mha.linear_v.weight"]).T)
                mha.linear_o.kernel.assign(_to_numpy(hf_state_dict[f"{a_pref}mha.linear_o.weight"]).T)
                mha.linear_o.bias.assign(_to_numpy(hf_state_dict[f"{a_pref}mha.linear_o.bias"]))
                if mha.linear_g is not None and f"{a_pref}mha.linear_g.weight" in hf_state_dict:
                    mha.linear_g.kernel.assign(_to_numpy(hf_state_dict[f"{a_pref}mha.linear_g.weight"]).T)
                    mha.linear_g.bias.assign(_to_numpy(hf_state_dict[f"{a_pref}mha.linear_g.bias"]))

        # mlp_pair
        if f"{prefix}mlp_pair.mlp.0.weight" in hf_state_dict:
            block.mlp_pair.layernorm.gamma.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_pair.mlp.0.weight"]))
            block.mlp_pair.layernorm.beta.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_pair.mlp.0.bias"]))
            block.mlp_pair.linear_1.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_pair.mlp.1.weight"]).T)
            block.mlp_pair.linear_1.bias.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_pair.mlp.1.bias"]))
            block.mlp_pair.linear_2.kernel.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_pair.mlp.3.weight"]).T)
            block.mlp_pair.linear_2.bias.assign(_to_numpy(hf_state_dict[f"{prefix}mlp_pair.mlp.3.bias"]))

    # 7. Structure Module
    sm = trunk.structure_module
    sm_pref = "trunk.structure_module."
    if f"{sm_pref}layer_norm_s.weight" in hf_state_dict:
        sm.layer_norm_s.gamma.assign(_to_numpy(hf_state_dict[f"{sm_pref}layer_norm_s.weight"]))
        sm.layer_norm_s.beta.assign(_to_numpy(hf_state_dict[f"{sm_pref}layer_norm_s.bias"]))
    if f"{sm_pref}layer_norm_z.weight" in hf_state_dict:
        sm.layer_norm_z.gamma.assign(_to_numpy(hf_state_dict[f"{sm_pref}layer_norm_z.weight"]))
        sm.layer_norm_z.beta.assign(_to_numpy(hf_state_dict[f"{sm_pref}layer_norm_z.bias"]))
    if f"{sm_pref}linear_in.weight" in hf_state_dict:
        sm.linear_in.kernel.assign(_to_numpy(hf_state_dict[f"{sm_pref}linear_in.weight"]).T)
        sm.linear_in.bias.assign(_to_numpy(hf_state_dict[f"{sm_pref}linear_in.bias"]))
    if f"{sm_pref}layer_norm_ipa.weight" in hf_state_dict:
        sm.layer_norm_ipa.gamma.assign(_to_numpy(hf_state_dict[f"{sm_pref}layer_norm_ipa.weight"]))
        sm.layer_norm_ipa.beta.assign(_to_numpy(hf_state_dict[f"{sm_pref}layer_norm_ipa.bias"]))

    # IPA
    ipa = sm.ipa
    ipa_pref = f"{sm_pref}ipa."
    if f"{ipa_pref}linear_q.weight" in hf_state_dict:
        ipa.linear_q.kernel.assign(_to_numpy(hf_state_dict[f"{ipa_pref}linear_q.weight"]).T)
        ipa.linear_kv.kernel.assign(_to_numpy(hf_state_dict[f"{ipa_pref}linear_kv.weight"]).T)
        ipa.linear_q_points.kernel.assign(_to_numpy(hf_state_dict[f"{ipa_pref}linear_q_points.weight"]).T)
        ipa.linear_kv_points.kernel.assign(_to_numpy(hf_state_dict[f"{ipa_pref}linear_kv_points.weight"]).T)
        ipa.linear_b.kernel.assign(_to_numpy(hf_state_dict[f"{ipa_pref}linear_b.weight"]).T)
        ipa.head_weights.assign(_to_numpy(hf_state_dict[f"{ipa_pref}head_weights"]))
        ipa.linear_out.kernel.assign(_to_numpy(hf_state_dict[f"{ipa_pref}linear_out.weight"]).T)
        ipa.linear_out.bias.assign(_to_numpy(hf_state_dict[f"{ipa_pref}linear_out.bias"]))

    # bb_update
    if f"{sm_pref}bb_update.linear.weight" in hf_state_dict:
        sm.bb_update.linear.kernel.assign(_to_numpy(hf_state_dict[f"{sm_pref}bb_update.linear.weight"]).T)
        sm.bb_update.linear.bias.assign(_to_numpy(hf_state_dict[f"{sm_pref}bb_update.linear.bias"]))

    # angle_resnet
    ar = sm.angle_resnet
    ar_pref = f"{sm_pref}angle_resnet."
    if f"{ar_pref}linear_in.weight" in hf_state_dict:
        ar.linear_in.kernel.assign(_to_numpy(hf_state_dict[f"{ar_pref}linear_in.weight"]).T)
        ar.linear_in.bias.assign(_to_numpy(hf_state_dict[f"{ar_pref}linear_in.bias"]))
        ar.linear_initial.kernel.assign(_to_numpy(hf_state_dict[f"{ar_pref}linear_initial.weight"]).T)
        ar.linear_initial.bias.assign(_to_numpy(hf_state_dict[f"{ar_pref}linear_initial.bias"]))
        ar.linear_out.kernel.assign(_to_numpy(hf_state_dict[f"{ar_pref}linear_out.weight"]).T)
        ar.linear_out.bias.assign(_to_numpy(hf_state_dict[f"{ar_pref}linear_out.bias"]))
        for j, b in enumerate(ar.blocks):
            b.linear_1.kernel.assign(_to_numpy(hf_state_dict[f"{ar_pref}layers.{j}.linear_1.weight"]).T)
            b.linear_1.bias.assign(_to_numpy(hf_state_dict[f"{ar_pref}layers.{j}.linear_1.bias"]))
            b.linear_2.kernel.assign(_to_numpy(hf_state_dict[f"{ar_pref}layers.{j}.linear_2.weight"]).T)
            b.linear_2.bias.assign(_to_numpy(hf_state_dict[f"{ar_pref}layers.{j}.linear_2.bias"]))

    # transition
    tr = sm.transition
    tr_pref = f"{sm_pref}transition."
    if f"{tr_pref}layer_norm.weight" in hf_state_dict:
        tr.layer_norm.gamma.assign(_to_numpy(hf_state_dict[f"{tr_pref}layer_norm.weight"]))
        tr.layer_norm.beta.assign(_to_numpy(hf_state_dict[f"{tr_pref}layer_norm.bias"]))
        for j, l in enumerate(tr.transition_layers):
            l.linear_1.kernel.assign(_to_numpy(hf_state_dict[f"{tr_pref}layers.{j}.linear_1.weight"]).T)
            l.linear_1.bias.assign(_to_numpy(hf_state_dict[f"{tr_pref}layers.{j}.linear_1.bias"]))
            l.linear_2.kernel.assign(_to_numpy(hf_state_dict[f"{tr_pref}layers.{j}.linear_2.weight"]).T)
            l.linear_2.bias.assign(_to_numpy(hf_state_dict[f"{tr_pref}layers.{j}.linear_2.bias"]))
            l.linear_3.kernel.assign(_to_numpy(hf_state_dict[f"{tr_pref}layers.{j}.linear_3.weight"]).T)
            l.linear_3.bias.assign(_to_numpy(hf_state_dict[f"{tr_pref}layers.{j}.linear_3.bias"]))

    # 8. Heads
    if "distogram_head.weight" in hf_state_dict:
        keras_model.distogram_head.kernel.assign(_to_numpy(hf_state_dict["distogram_head.weight"]).T)
        keras_model.distogram_head.bias.assign(_to_numpy(hf_state_dict["distogram_head.bias"]))
    if "ptm_head.weight" in hf_state_dict:
        keras_model.ptm_head.kernel.assign(_to_numpy(hf_state_dict["ptm_head.weight"]).T)
        keras_model.ptm_head.bias.assign(_to_numpy(hf_state_dict["ptm_head.bias"]))
    if "lm_head.weight" in hf_state_dict:
        keras_model.lm_head.kernel.assign(_to_numpy(hf_state_dict["lm_head.weight"]).T)
        keras_model.lm_head.bias.assign(_to_numpy(hf_state_dict["lm_head.bias"]))

    if "lddt_head.0.weight" in hf_state_dict:
        keras_model.lddt_head.layer_norm.gamma.assign(_to_numpy(hf_state_dict["lddt_head.0.weight"]))
        keras_model.lddt_head.layer_norm.beta.assign(_to_numpy(hf_state_dict["lddt_head.0.bias"]))
        keras_model.lddt_head.linear_1.kernel.assign(_to_numpy(hf_state_dict["lddt_head.1.weight"]).T)
        keras_model.lddt_head.linear_1.bias.assign(_to_numpy(hf_state_dict["lddt_head.1.bias"]))
        keras_model.lddt_head.linear_2.kernel.assign(_to_numpy(hf_state_dict["lddt_head.2.weight"]).T)
        keras_model.lddt_head.linear_2.bias.assign(_to_numpy(hf_state_dict["lddt_head.2.bias"]))
        keras_model.lddt_head.linear_3.kernel.assign(_to_numpy(hf_state_dict["lddt_head.3.weight"]).T)
        keras_model.lddt_head.linear_3.bias.assign(_to_numpy(hf_state_dict["lddt_head.3.bias"]))

    return keras_model


def load_esmfold_from_hf(
    model_id_or_path: str = "facebook/esmfold_v1",
    config: Optional[ESMFoldConfig] = None,
) -> ESMFoldModel:
    """
    Instantiates an ESMFoldModel and loads weights directly from Hugging Face Hub or local path.

    Args:
        model_id_or_path: Hugging Face model repository ID (e.g. "facebook/esmfold_v1") or local path.
        config: Optional ESMFoldConfig. If None, loaded from pretrained config.

    Returns:
        ESMFoldModel with pretrained weights.
    """
    from transformers import AutoConfig, EsmForProteinFolding
    import torch

    hf_config = AutoConfig.from_pretrained(model_id_or_path)
    if config is None:
        config = ESMFoldConfig.from_hf(hf_config.to_dict())

    keras_model = ESMFoldModel(config)

    # Initialize model variables with dummy input
    dummy_input = ops.convert_to_tensor(np.array([[0, 1, 2, 3]], dtype=np.int32))
    keras_model(dummy_input)

    # Load state dict
    try:
        hf_model = EsmForProteinFolding.from_pretrained(model_id_or_path)
        state_dict = hf_model.state_dict()
    except Exception:
        from huggingface_hub import hf_hub_download
        weight_file = hf_hub_download(repo_id=model_id_or_path, filename="pytorch_model.bin")
        state_dict = torch.load(weight_file, map_location="cpu")

    convert_esmfold_weights_from_hf(state_dict, keras_model)
    return keras_model

