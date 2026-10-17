"""
Configuration classes for ESMFold protein structure models.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


@dataclass
class StructureModuleConfig:
    """Configuration for the Invariant Point Attention (IPA) Structure Module."""
    sequence_dim: int = 384
    pairwise_dim: int = 128
    ipa_dim: int = 16
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    num_blocks: int = 8
    num_angles: int = 7
    num_resnet_blocks: int = 2
    num_transition_layers: int = 1
    resnet_dim: int = 128
    trans_scale_factor: float = 10.0
    epsilon: float = 1e-8
    inf: float = 1e5
    dropout_rate: float = 0.0


@dataclass
class FoldingTrunkConfig:
    """Configuration for the Evoformer-style Folding Trunk."""
    num_blocks: int = 48
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    max_recycles: int = 4
    recycle_bins: int = 15
    dropout: float = 0.0
    structure_module: StructureModuleConfig = field(default_factory=StructureModuleConfig)


@dataclass
class ESMFoldConfig:
    """
    Configuration for full ESMFold end-to-end structure prediction model.
    """
    # ESM-2 Backbone configuration
    esm_vocab_size: int = 33
    esm_hidden_size: int = 2560
    esm_num_layers: int = 36
    esm_num_heads: int = 40
    esm_intermediate_size: int = 10240
    esm_type: str = "esm2_3B"

    # Folding Trunk configuration
    trunk: FoldingTrunkConfig = field(default_factory=FoldingTrunkConfig)

    # Heads & Features
    lddt_head_hid_dim: int = 128
    distogram_bins: int = 64
    lddt_bins: int = 50
    embed_aa: bool = True
    bypass_lm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ESMFoldConfig":
        data = dict(data)
        valid = {f.name for f in fields(cls)}
        if "trunk" in data and isinstance(data["trunk"], dict):
            trunk_dict = dict(data["trunk"])
            trunk_valid = {f.name for f in fields(FoldingTrunkConfig)}
            if "structure_module" in trunk_dict and isinstance(trunk_dict["structure_module"], dict):
                sm_valid = {f.name for f in fields(StructureModuleConfig)}
                sm_dict = {k: v for k, v in trunk_dict["structure_module"].items() if k in sm_valid}
                trunk_dict["structure_module"] = StructureModuleConfig(**sm_dict)
            trunk_dict = {k: v for k, v in trunk_dict.items() if k in trunk_valid}
            data["trunk"] = FoldingTrunkConfig(**trunk_dict)
        data = {k: v for k, v in data.items() if k in valid}
        return cls(**data)

    @classmethod
    def from_preset(cls, preset_name: str) -> "ESMFoldConfig":
        presets = {
            "esmfold_v1": {
                "esm_vocab_size": 33,
                "esm_hidden_size": 2560,
                "esm_num_layers": 36,
                "esm_num_heads": 40,
                "esm_intermediate_size": 10240,
                "esm_type": "esm2_3B",
                "trunk": {
                    "num_blocks": 48,
                    "sequence_state_dim": 1024,
                    "pairwise_state_dim": 128,
                    "sequence_head_width": 32,
                    "pairwise_head_width": 32,
                    "position_bins": 32,
                    "max_recycles": 4,
                    "recycle_bins": 15,
                    "dropout": 0.0,
                    "structure_module": {
                        "sequence_dim": 384,
                        "pairwise_dim": 128,
                        "ipa_dim": 16,
                        "num_heads_ipa": 12,
                        "num_qk_points": 4,
                        "num_v_points": 8,
                        "num_blocks": 8,
                        "num_angles": 7,
                        "num_resnet_blocks": 2,
                        "num_transition_layers": 1,
                        "resnet_dim": 128,
                        "trans_scale_factor": 10.0,
                        "epsilon": 1e-8,
                        "inf": 1e5,
                        "dropout_rate": 0.0,
                    },
                },
                "lddt_head_hid_dim": 128,
                "distogram_bins": 64,
                "lddt_bins": 50,
                "embed_aa": True,
                "bypass_lm": False,
            },
            "esmfold_tiny": {
                "esm_vocab_size": 33,
                "esm_hidden_size": 64,
                "esm_num_layers": 2,
                "esm_num_heads": 4,
                "esm_intermediate_size": 128,
                "esm_type": "esm2_tiny",
                "trunk": {
                    "num_blocks": 2,
                    "sequence_state_dim": 64,
                    "pairwise_state_dim": 32,
                    "sequence_head_width": 16,
                    "pairwise_head_width": 16,
                    "position_bins": 16,
                    "max_recycles": 1,
                    "recycle_bins": 15,
                    "dropout": 0.0,
                    "structure_module": {
                        "sequence_dim": 64,
                        "pairwise_dim": 32,
                        "ipa_dim": 16,
                        "num_heads_ipa": 4,
                        "num_qk_points": 4,
                        "num_v_points": 4,
                        "num_blocks": 2,
                        "num_angles": 7,
                        "num_resnet_blocks": 1,
                        "num_transition_layers": 1,
                        "resnet_dim": 32,
                        "trans_scale_factor": 10.0,
                        "epsilon": 1e-8,
                        "inf": 1e5,
                        "dropout_rate": 0.0,
                    },
                },
                "lddt_head_hid_dim": 32,
                "distogram_bins": 64,
                "lddt_bins": 50,
                "embed_aa": True,
                "bypass_lm": False,
            },
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")

        return cls.from_dict(presets[preset_name])

    @classmethod
    def from_hf(cls, hf_config_dict: dict[str, Any]) -> "ESMFoldConfig":
        """Extracts ESMFoldConfig from a Hugging Face AutoConfig dictionary."""
        cfg_d = dict(hf_config_dict)
        esm_d = cfg_d.get("esmfold_config", {})
        if not isinstance(esm_d, dict):
            esm_d = getattr(esm_d, "to_dict", lambda: {})()

        trunk_d = esm_d.get("trunk", {})
        if not isinstance(trunk_d, dict):
            trunk_d = getattr(trunk_d, "to_dict", lambda: {})()

        sm_d = trunk_d.get("structure_module", {})
        if not isinstance(sm_d, dict):
            sm_d = getattr(sm_d, "to_dict", lambda: {})()

        sm_cfg = StructureModuleConfig(
            sequence_dim=sm_d.get("sequence_dim", 384),
            pairwise_dim=sm_d.get("pairwise_dim", 128),
            ipa_dim=sm_d.get("ipa_dim", 16),
            num_heads_ipa=sm_d.get("num_heads_ipa", 12),
            num_qk_points=sm_d.get("num_qk_points", 4),
            num_v_points=sm_d.get("num_v_points", 8),
            num_blocks=sm_d.get("num_blocks", 8),
            num_angles=sm_d.get("num_angles", 7),
            num_resnet_blocks=sm_d.get("num_resnet_blocks", 2),
            num_transition_layers=sm_d.get("num_transition_layers", 1),
            resnet_dim=sm_d.get("resnet_dim", 128),
            trans_scale_factor=float(sm_d.get("trans_scale_factor", 10.0)),
            epsilon=float(sm_d.get("epsilon", 1e-8)),
            inf=float(sm_d.get("inf", 1e5)),
            dropout_rate=float(sm_d.get("dropout_rate", 0.0)),
        )

        trunk_cfg = FoldingTrunkConfig(
            num_blocks=trunk_d.get("num_blocks", 48),
            sequence_state_dim=trunk_d.get("sequence_state_dim", 1024),
            pairwise_state_dim=trunk_d.get("pairwise_state_dim", 128),
            sequence_head_width=trunk_d.get("sequence_head_width", 32),
            pairwise_head_width=trunk_d.get("pairwise_head_width", 32),
            position_bins=trunk_d.get("position_bins", 32),
            max_recycles=trunk_d.get("max_recycles", 4),
            recycle_bins=trunk_d.get("recycle_bins", 15),
            dropout=float(trunk_d.get("dropout", 0.0)),
            structure_module=sm_cfg,
        )

        return cls(
            esm_vocab_size=cfg_d.get("vocab_size", 33),
            esm_hidden_size=cfg_d.get("hidden_size", 2560),
            esm_num_layers=cfg_d.get("num_hidden_layers", 36),
            esm_num_heads=cfg_d.get("num_attention_heads", 40),
            esm_intermediate_size=cfg_d.get("intermediate_size", 10240),
            esm_type=esm_d.get("esm_type", "esm2_3B"),
            trunk=trunk_cfg,
            lddt_head_hid_dim=esm_d.get("lddt_head_hid_dim", 128),
            distogram_bins=cfg_d.get("distogram_bins", 64),
            lddt_bins=cfg_d.get("lddt_bins", 50),
            embed_aa=esm_d.get("embed_aa", True),
            bypass_lm=esm_d.get("bypass_lm", False),
        )

