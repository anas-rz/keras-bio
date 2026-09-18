"""Exact numerical parity verification for ESM-2 against Hugging Face PyTorch."""

import keras.ops as ops
import numpy as np
import pytest
import torch
from transformers import EsmConfig as HFEsmConfig
from transformers import EsmForMaskedLM as HFEsmForMaskedLM
from transformers import EsmModel as HFEsmModel

from keras_bio.protein.esm2.config import ESM2Config
from keras_bio.protein.esm2.convert import convert_esm2_weights_from_hf
from keras_bio.protein.esm2.modeling import ESM2ForMaskedLM, ESM2Model


def test_esm2_encoder_parity_synthetic():
    """Verify ESM2Model numerical parity against HF EsmModel."""
    torch.manual_seed(42)
    np.random.seed(42)

    hf_cfg = HFEsmConfig(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        pad_token_id=1,
        mask_token_id=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        position_embedding_type="rotary",
        token_dropout=True,
    )
    hf_cfg._attn_implementation = "eager"
    torch_model = HFEsmModel(hf_cfg).eval()

    k_cfg = ESM2Config.from_hf(hf_cfg.to_dict())
    k_model = ESM2Model(k_cfg)
    k_model.build((None, 12))

    convert_esm2_weights_from_hf(torch_model.state_dict(), k_model)

    input_ids = np.array([[0, 4, 5, 6, 7, 2, 1, 1], [0, 8, 9, 10, 11, 12, 13, 2]], dtype="int32")
    attention_mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype="int32")

    with torch.no_grad():
        pt_out = torch_model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        ).last_hidden_state.numpy()

    k_out = k_model({"input_ids": input_ids, "attention_mask": attention_mask})["last_hidden_state"]
    k_out_np = ops.convert_to_numpy(k_out)

    diff = np.max(np.abs(pt_out - k_out_np))
    assert diff < 1e-4, f"ESM2 encoder parity max diff too large: {diff}"


def test_esm2_contact_prediction_parity_synthetic():
    """Verify ESM2Model.predict_contacts matches HF EsmModel.predict_contacts."""
    torch.manual_seed(42)
    np.random.seed(42)

    hf_cfg = HFEsmConfig(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        pad_token_id=1,
        mask_token_id=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        position_embedding_type="rotary",
        token_dropout=True,
    )
    hf_cfg._attn_implementation = "eager"
    torch_model = HFEsmModel(hf_cfg).eval()

    k_cfg = ESM2Config.from_hf(hf_cfg.to_dict())
    k_model = ESM2Model(k_cfg)
    k_model.build((None, 12))

    convert_esm2_weights_from_hf(torch_model.state_dict(), k_model)

    input_ids = np.array([[0, 4, 5, 6, 7, 2, 1, 1], [0, 8, 9, 10, 11, 12, 13, 2]], dtype="int32")
    attention_mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype="int32")

    with torch.no_grad():
        pt_contacts = torch_model.predict_contacts(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        ).numpy()

    k_contacts = k_model.predict_contacts(input_ids, attention_mask=attention_mask)
    k_contacts_np = ops.convert_to_numpy(k_contacts)

    diff = np.max(np.abs(pt_contacts - k_contacts_np))
    assert diff < 1e-4, f"ESM2 contact prediction parity max diff too large: {diff}"


def test_esm2_masked_lm_parity_synthetic():
    """Verify ESM2ForMaskedLM matches HF EsmForMaskedLM."""
    torch.manual_seed(42)
    np.random.seed(42)

    hf_cfg = HFEsmConfig(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        pad_token_id=1,
        mask_token_id=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        position_embedding_type="rotary",
        token_dropout=True,
    )
    hf_cfg._attn_implementation = "eager"
    torch_model = HFEsmForMaskedLM(hf_cfg).eval()

    k_cfg = ESM2Config.from_hf(hf_cfg.to_dict())
    k_model = ESM2ForMaskedLM(k_cfg)
    k_model.build((None, 12))

    convert_esm2_weights_from_hf(torch_model.state_dict(), k_model)

    input_ids = np.array([[0, 4, 32, 6, 7, 2]], dtype="int32")
    attention_mask = np.ones((1, 6), dtype="int32")

    with torch.no_grad():
        pt_logits = torch_model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        ).logits.numpy()

    k_logits = k_model({"input_ids": input_ids, "attention_mask": attention_mask})["logits"]
    k_logits_np = ops.convert_to_numpy(k_logits)

    diff = np.max(np.abs(pt_logits - k_logits_np))
    assert diff < 5e-4, f"ESM2 Masked LM logits parity max diff too large: {diff}"


def test_esm2_pretrained_checkpoint_parity():
    """Verify parity against official facebook/esm2_t6_8M_UR50D checkpoint."""
    try:
        from transformers import AutoTokenizer, EsmForMaskedLM

        hf_model = EsmForMaskedLM.from_pretrained(
            "facebook/esm2_t6_8M_UR50D", attn_implementation="eager"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    except Exception as e:
        pytest.skip(f"Could not load checkpoint: {e}")

    k_cfg = ESM2Config.from_hf(hf_model.config.to_dict())
    k_model = ESM2ForMaskedLM(k_cfg)
    k_model.build((None, 16))

    convert_esm2_weights_from_hf(hf_model.state_dict(), k_model)

    inputs = tokenizer(["MKTVRQERLKSIVR"], padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()

    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits.numpy()
        hf_contacts = hf_model.esm.predict_contacts(
            inputs["input_ids"], inputs["attention_mask"]
        ).numpy()

    k_out = k_model({"input_ids": input_ids, "attention_mask": attention_mask})
    k_logits = ops.convert_to_numpy(k_out["logits"])
    k_contacts = ops.convert_to_numpy(k_model.esm.predict_contacts(input_ids, attention_mask=attention_mask))

    logits_diff = np.max(np.abs(hf_logits - k_logits))
    contacts_diff = np.max(np.abs(hf_contacts - k_contacts))

    assert logits_diff < 5e-4, f"Pretrained logits diff too large: {logits_diff}"
    assert contacts_diff < 5e-4, f"Pretrained contacts diff too large: {contacts_diff}"
