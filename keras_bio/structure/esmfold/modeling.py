"""
ESMFold protein structure prediction model.

Integrates ESM-2 language model backbone with Evoformer Folding Trunk,
Invariant Point Attention (IPA) Structure Module, confidence prediction heads,
and PDB structure output formatting.
"""

from typing import Any, Optional, Union
import numpy as np

import keras
from keras import layers, ops

import keras_bio.structure.esmfold.geometry as geometry
import keras_bio.structure.esmfold.residue_constants as rc
import keras_bio.structure.esmfold.feats as feats
import keras_bio.structure.esmfold.layers as kl
import keras_bio.structure.esmfold.pdb_utils as pdb_utils
from keras_bio.structure.esmfold.config import ESMFoldConfig
from keras_bio.protein.esm2.config import ESM2Config
from keras_bio.protein.esm2.modeling import ESM2Model


def compute_tm(logits, max_bin: int = 31, no_bins: int = 64):
    """
    Computes predicted TM-score from pairwise distogram / PTM logits.
    """
    boundaries = ops.linspace(0.0, float(max_bin), no_bins - 1)
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2.0
    bin_centers = ops.concatenate([bin_centers, ops.expand_dims(bin_centers[-1] + step, axis=-1)], axis=0)

    n = ops.shape(logits)[-2]
    clipped_n = max(n, 19)
    d0 = 1.24 * (float(clipped_n) - 15.0) ** (1.0 / 3.0) - 1.8

    probs = ops.softmax(logits, axis=-1)
    tm_per_bin = 1.0 / (1.0 + (ops.square(bin_centers)) / (d0**2))
    predicted_tm_term = ops.sum(probs * tm_per_bin, axis=-1)
    per_alignment = ops.mean(predicted_tm_term, axis=-1)
    return ops.max(per_alignment, axis=-1)


def compute_predicted_aligned_error(logits, max_bin: int = 31, no_bins: int = 64):
    """
    Computes Predicted Aligned Error (PAE) matrix in Angstroms from PTM logits.
    """
    boundaries = ops.linspace(0.0, float(max_bin), no_bins - 1)
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2.0
    bin_centers = ops.concatenate([bin_centers, ops.expand_dims(bin_centers[-1] + step, axis=-1)], axis=0)

    probs = ops.softmax(logits, axis=-1)
    pae = ops.sum(probs * bin_centers, axis=-1)
    max_pae = bin_centers[-1]
    return {
        "aligned_confidence_probs": probs,
        "predicted_aligned_error": pae,
        "max_predicted_aligned_error": max_pae,
    }


@keras.saving.register_keras_serializable(package="keras_bio")
class ESMFoldModel(keras.Model):
    """
    Complete end-to-end ESMFold 3D protein structure prediction model.
    """

    def __init__(self, config: ESMFoldConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # ESM-2 Backbone
        esm_cfg = ESM2Config(
            vocab_size=config.esm_vocab_size,
            hidden_size=config.esm_hidden_size,
            num_hidden_layers=config.esm_num_layers,
            num_attention_heads=config.esm_num_heads,
            intermediate_size=config.esm_intermediate_size,
            token_dropout=False,
        )
        self.esm = ESM2Model(esm_cfg, name="esm")

        # Layer combination parameter for ESM hidden states
        self.esm_s_combine = self.add_weight(
            name="esm_s_combine",
            shape=(config.esm_num_layers + 1,),
            initializer="zeros",
            trainable=True,
        )

        # Mapping table from AF2 aatype (0..20) to ESM token IDs
        # 0 is padding in aatype+1
        esm_reorder = [1] + [
            ESMFoldModel._get_esm_token_id(v) for v in rc.restypes_with_x
        ]
        self.af2_to_esm = ops.convert_to_tensor(esm_reorder, dtype="int32")

        # Sequence feature projection MLP
        self.esm_s_mlp_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="esm_s_mlp_0")
        self.esm_s_mlp_dense1 = layers.Dense(config.trunk.sequence_state_dim, use_bias=True, name="esm_s_mlp_1")
        self.esm_s_mlp_relu = layers.Activation("relu")
        self.esm_s_mlp_dense2 = layers.Dense(config.trunk.sequence_state_dim, use_bias=True, name="esm_s_mlp_3")

        # Optional amino acid embedding table (0..20 amino acids, 21 unk, 22 mask)
        if config.embed_aa:
            self.embedding = layers.Embedding(
                input_dim=23,
                output_dim=config.trunk.sequence_state_dim,
                name="embedding",
            )
        else:
            self.embedding = None

        # Evoformer Folding Trunk
        self.trunk = kl.EsmFoldingTrunk(config.trunk, name="trunk")

        # Confidence & Metric Prediction Heads
        self.distogram_head = layers.Dense(config.distogram_bins, use_bias=True, name="distogram_head")
        self.ptm_head = layers.Dense(config.distogram_bins, use_bias=True, name="ptm_head")
        self.lm_head = layers.Dense(23, use_bias=True, name="lm_head")
        self.lddt_head = kl.LDDTHead(
            config.trunk.structure_module.sequence_dim,
            config.lddt_head_hid_dim,
            config.lddt_bins,
            name="lddt_head",
        )

    @staticmethod
    def _get_esm_token_id(aa_char: str) -> int:
        """Standard ESM-2 vocabulary token mapping for amino acids."""
        esm_vocab = {
            "<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3,
            "L": 4, "A": 5, "G": 6, "V": 7, "S": 8, "E": 9,
            "R": 10, "T": 11, "I": 12, "D": 13, "P": 14, "K": 15,
            "Q": 16, "N": 17, "F": 18, "Y": 19, "M": 20, "H": 21,
            "W": 22, "C": 23, "X": 24, "B": 25, "U": 26, "Z": 27,
            "O": 28, ".": 29, "-": 30, "<null_1>": 31, "<mask>": 32,
        }
        return esm_vocab.get(aa_char, 3)

    def af2_idx_to_esm_idx(self, aa, mask):
        """Maps AF2 aatype indices (0..20) to ESM token IDs, handling masked positions."""
        # Shift by 1: 0 becomes 1 (<pad>) when masked
        shifted = aa + 1
        shifted = ops.where(mask > 0, shifted, ops.zeros_like(shifted))
        return ops.take(self.af2_to_esm, shifted, axis=0)

    def compute_language_model_representations(self, esmaa, attention_mask):
        batch_dim = ops.shape(esmaa)[0]
        seq_len = ops.shape(esmaa)[1]

        if self.config.bypass_lm:
            return ops.zeros(
                (batch_dim, seq_len, self.config.esm_num_layers + 1, self.config.esm_hidden_size),
                dtype="float32",
            )

        # Prepend BOS (<cls>=0), pad extra column, place EOS (<eos>=2) at valid length
        bos = ops.zeros((batch_dim, 1), dtype="int32")
        pad_col = ops.ones((batch_dim, 1), dtype="int32")
        esmaa_padded = ops.concatenate([bos, esmaa, pad_col], axis=1)

        eos_indices = ops.cast(ops.sum(attention_mask, axis=-1), "int32") + 1
        seq_indices = ops.arange(seq_len + 2, dtype="int32")[None, :]

        eos_mask = ops.equal(seq_indices, ops.expand_dims(eos_indices, -1))
        pad_mask = ops.greater(seq_indices, ops.expand_dims(eos_indices, -1))

        esmaa_padded = ops.where(eos_mask, ops.ones_like(esmaa_padded) * 2, esmaa_padded)
        esmaa_padded = ops.where(pad_mask, ops.ones_like(esmaa_padded) * 1, esmaa_padded)

        attn_mask_padded = ops.cast(
            ops.less_equal(seq_indices, ops.expand_dims(eos_indices, -1)),
            attention_mask.dtype,
        )

        # Forward pass through ESM-2 language model backbone
        esm_out = self.esm(esmaa_padded, attention_mask=attn_mask_padded, output_hidden_states=True)
        hidden_states = esm_out["hidden_states"]

        # Stack representations: [B, L+2, num_layers + 1, hidden_size]
        esm_s = ops.stack(hidden_states, axis=2)
        # Strip BOS and EOS tokens -> [B, L, num_layers + 1, hidden_size]
        esm_s = esm_s[:, 1:-1, :, :]
        return esm_s

    def call(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        position_ids: Optional[Any] = None,
        num_recycles: Optional[int] = None,
        training: Optional[bool] = None,
    ) -> dict[str, Any]:
        """
        Forward pass for ESMFold.

        Args:
            input_ids: [B, L] AF2 aatype amino acid indices (0..20).
            attention_mask: [B, L] binary mask indicating valid residue positions (1=valid, 0=padding).
            position_ids: [B, L] residue sequence positions (default 0..L-1).
            num_recycles: Number of recycling iterations (default config.trunk.max_recycles).

        Returns:
            Dictionary containing 3D atomic positions, frames, pLDDT, and confidence metrics.
        """
        aa = input_ids
        batch_dim = ops.shape(aa)[0]
        seq_len = ops.shape(aa)[1]

        if attention_mask is None:
            attention_mask = ops.ones_like(aa, dtype="float32")
        else:
            attention_mask = ops.cast(attention_mask, dtype="float32")
        if position_ids is None:
            position_ids = ops.broadcast_to(ops.arange(seq_len, dtype="int32"), (batch_dim, seq_len))

        # 1. ESM language model representations
        esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)
        esm_s = self.compute_language_model_representations(esmaa, attention_mask)

        # Softmax combination across layers
        combine_weights = ops.softmax(self.esm_s_combine, axis=0)
        # [1, 1, num_layers+1, 1]
        weight_shape = (1, 1, self.config.esm_num_layers + 1, 1)
        esm_s = ops.sum(esm_s * ops.reshape(combine_weights, weight_shape), axis=2)

        # 2. Sequence projection MLP
        s_s_0 = self.esm_s_mlp_norm(esm_s)
        s_s_0 = self.esm_s_mlp_dense1(s_s_0)
        s_s_0 = self.esm_s_mlp_relu(s_s_0)
        s_s_0 = self.esm_s_mlp_dense2(s_s_0)

        if self.embedding is not None:
            s_s_0 = s_s_0 + self.embedding(aa)

        # 3. Initialize pairwise representations
        s_z_0 = ops.zeros(
            (batch_dim, seq_len, seq_len, self.config.trunk.pairwise_state_dim),
            dtype=s_s_0.dtype,
        )

        # 4. Evoformer Folding Trunk
        structure = self.trunk(
            s_s_0,
            s_z_0,
            aa,
            position_ids,
            mask=attention_mask,
            no_recycles=num_recycles,
        )

        # 5. Distogram head
        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + ops.swapaxes(disto_logits, 1, 2)) / 2.0
        structure["distogram_logits"] = disto_logits

        # 6. LM logits
        structure["lm_logits"] = self.lm_head(structure["s_s"])

        # 7. Atom masks & index mappings
        atom_masks = feats.make_atom14_masks(aa)
        mask_expanded = ops.expand_dims(attention_mask, axis=-1)
        structure["atom14_atom_exists"] = atom_masks["atom14_atom_exists"] * ops.cast(mask_expanded, "float32")
        structure["atom37_atom_exists"] = atom_masks["atom37_atom_exists"] * ops.cast(mask_expanded, "float32")
        structure["residx_atom14_to_atom37"] = atom_masks["residx_atom14_to_atom37"]
        structure["residx_atom37_to_atom14"] = atom_masks["residx_atom37_to_atom14"]
        structure["residue_index"] = position_ids
        structure["aatype"] = aa

        # 8. Confidence heads (pLDDT, PTM, PAE)
        lddt_logits = self.lddt_head(structure["states"])
        structure["lddt_head"] = lddt_logits
        # Final block pLDDT
        plddt = kl.categorical_lddt(lddt_logits[:, -1], bins=self.config.lddt_bins)
        structure["plddt"] = plddt

        ptm_logits = self.ptm_head(structure["s_z"])
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = compute_tm(ptm_logits, no_bins=self.config.distogram_bins)
        structure.update(compute_predicted_aligned_error(ptm_logits, no_bins=self.config.distogram_bins))

        # Final atom37 coordinates
        structure["atom37_positions"] = feats.atom14_to_atom37(
            structure["positions"][:, -1],
            structure["residx_atom37_to_atom14"],
            structure["atom37_atom_exists"],
        )

        return structure

    def infer(
        self,
        sequences: Union[str, list[str]],
        num_recycles: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Runs protein structure prediction directly on amino acid sequence string(s).

        Args:
            sequences: Protein amino acid sequence string (e.g. "MLKNVQVQLV") or list of strings.
            num_recycles: Number of recycles (defaults to config.trunk.max_recycles).

        Returns:
            Dictionary containing 3D atomic coordinates, pLDDT scores, and PDB formatted strings.
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        max_len = max(len(s) for s in seq_list)
        batch_size = len(seq_list)

        aatype_np = np.zeros((batch_size, max_len), dtype=np.int32)
        mask_np = np.zeros((batch_size, max_len), dtype=np.float32)

        mapping = rc.restype_order_with_x
        for i, seq in enumerate(seq_list):
            for j, char in enumerate(seq):
                aatype_np[i, j] = mapping.get(char, 20)
                mask_np[i, j] = 1.0

        input_ids = ops.convert_to_tensor(aatype_np)
        attention_mask = ops.convert_to_tensor(mask_np)

        outputs = self(input_ids, attention_mask=attention_mask, num_recycles=num_recycles)
        outputs["pdbs"] = self.output_to_pdb(outputs)
        return outputs

    @staticmethod
    def output_to_pdb(outputs: dict[str, Any]) -> list[str]:
        """
        Converts model output dictionary into standard PDB text strings.

        Args:
            outputs: Dictionary returned by model forward pass.

        Returns:
            List of standard PDB formatted strings (one per sequence in batch).
        """
        atom37_pos = ops.convert_to_numpy(outputs["atom37_positions"])
        atom37_mask = ops.convert_to_numpy(outputs["atom37_atom_exists"])
        aatype = ops.convert_to_numpy(outputs["aatype"])
        residue_idx = ops.convert_to_numpy(outputs["residue_index"])
        plddt = ops.convert_to_numpy(outputs["plddt"])

        batch_size = aatype.shape[0]
        pdb_strings = []

        for b in range(batch_size):
            prot = pdb_utils.ProteinStructure(
                aatype=aatype[b],
                atom_positions=atom37_pos[b],
                atom_mask=atom37_mask[b],
                residue_index=residue_idx[b] + 1,  # 1-indexed for PDB
                b_factors=plddt[b],
                chain_tag="A",
            )
            pdb_str = pdb_utils.protein_to_pdb(prot)
            pdb_strings.append(pdb_str)

        return pdb_strings

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(self.config.to_dict())
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ESMFoldModel":
        cfg = ESMFoldConfig.from_dict(config)
        kwargs = {k: v for k, v in config.items() if k not in cfg.to_dict()}
        model = cls(config=cfg, **kwargs)
        # Warmup dummy pass so all sublayer weights exist for weight loading
        dummy_x = ops.zeros((1, 4), dtype="int32")
        model(dummy_x, num_recycles=0)
        return model

