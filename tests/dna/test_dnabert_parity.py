"""Exact numerical parity verification for DNABERT against Hugging Face PyTorch."""

import keras.ops as ops
import numpy as np
import pytest
import torch
from transformers import BertConfig as HFBertConfig
from transformers import BertForMaskedLM as HFBertForMaskedLM
from transformers import BertModel as HFBertModel

from keras_bio.dna.dnabert.config import DNABERTConfig
from keras_bio.dna.dnabert.convert import convert_dnabert_weights_from_hf
from keras_bio.dna.dnabert.modeling import DNABERTForMaskedLM, DNABERTModel
from keras_bio.dna.dnabert.tokenizer import DNABERTTokenizer


def test_dnabert_encoder_parity_synthetic():
    """Verify DNABERTModel numerical parity against HF BertModel."""
    torch.manual_seed(42)
    np.random.seed(42)

    hf_cfg = HFBertConfig(
        vocab_size=69,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    torch_model = HFBertModel(hf_cfg).eval()

    k_cfg = DNABERTConfig.from_hf(hf_cfg.to_dict())
    k_model = DNABERTModel(k_cfg)
    k_model.build((None, 12))

    convert_dnabert_weights_from_hf(torch_model.state_dict(), k_model)

    input_ids = np.array([[2, 10, 15, 20, 3, 0], [2, 5, 8, 9, 12, 3]], dtype="int32")
    attention_mask = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype="int32")
    token_type_ids = np.zeros_like(input_ids)

    with torch.no_grad():
        pt_out = torch_model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
        )
        pt_hidden = pt_out.last_hidden_state.numpy()
        pt_pooler = pt_out.pooler_output.numpy()

    k_out = k_model({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    k_hidden = ops.convert_to_numpy(k_out["last_hidden_state"])
    k_pooler = ops.convert_to_numpy(k_out["pooler_output"])

    diff_hidden = np.max(np.abs(pt_hidden - k_hidden))
    diff_pooler = np.max(np.abs(pt_pooler - k_pooler))

    assert diff_hidden < 1e-4, f"DNABERT encoder hidden state max diff too large: {diff_hidden}"
    assert diff_pooler < 1e-4, f"DNABERT encoder pooler output max diff too large: {diff_pooler}"


def test_dnabert_masked_lm_parity_synthetic():
    """Verify DNABERTForMaskedLM matches HF BertForMaskedLM."""
    torch.manual_seed(42)
    np.random.seed(42)

    hf_cfg = HFBertConfig(
        vocab_size=69,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    torch_model = HFBertForMaskedLM(hf_cfg).eval()

    k_cfg = DNABERTConfig.from_hf(hf_cfg.to_dict())
    k_model = DNABERTForMaskedLM(k_cfg)
    k_model.build((None, 12))

    convert_dnabert_weights_from_hf(torch_model.state_dict(), k_model)

    input_ids = np.array([[2, 10, 4, 20, 3]], dtype="int32")
    attention_mask = np.ones((1, 5), dtype="int32")

    with torch.no_grad():
        pt_logits = torch_model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        ).logits.numpy()

    k_logits = ops.convert_to_numpy(
        k_model({"input_ids": input_ids, "attention_mask": attention_mask})["logits"]
    )

    diff = np.max(np.abs(pt_logits - k_logits))
    assert diff < 2e-4, f"DNABERT Masked LM logits parity max diff too large: {diff}"


def test_dnabert_pretrained_checkpoint_parity():
    """Verify parity against official zhihan1996/DNA_bert_6 checkpoint."""
    try:
        from transformers import BertForMaskedLM, BertModel, BertTokenizer

        hf_bert = BertModel.from_pretrained("zhihan1996/DNA_bert_6").eval()
        hf_mlm = BertForMaskedLM.from_pretrained("zhihan1996/DNA_bert_6").eval()
        hf_tok = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    except Exception as e:
        pytest.skip(f"Could not load checkpoint: {e}")

    k_cfg = DNABERTConfig.from_hf(hf_bert.config.to_dict())
    k_bert = DNABERTModel(k_cfg)
    k_bert.build((None, 16))
    convert_dnabert_weights_from_hf(hf_bert.state_dict(), k_bert)

    k_mlm = DNABERTForMaskedLM(k_cfg)
    k_mlm.build((None, 16))
    convert_dnabert_weights_from_hf(hf_mlm.state_dict(), k_mlm)

    tok = DNABERTTokenizer(kmer=6)
    seq = "ATCGATCGATCGATCG"
    enc = tok(seq, return_tensors="np")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        hf_inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        hf_bert_out = hf_bert(**hf_inputs)
        hf_hidden = hf_bert_out.last_hidden_state.numpy()
        hf_pooler = hf_bert_out.pooler_output.numpy()
        hf_logits = hf_mlm(**hf_inputs).logits.numpy()

    k_out = k_bert({"input_ids": input_ids, "attention_mask": attention_mask})
    k_hidden = ops.convert_to_numpy(k_out["last_hidden_state"])
    k_pooler = ops.convert_to_numpy(k_out["pooler_output"])

    k_mlm_out = k_mlm({"input_ids": input_ids, "attention_mask": attention_mask})
    k_logits = ops.convert_to_numpy(k_mlm_out["logits"])

    diff_hidden = np.max(np.abs(hf_hidden - k_hidden))
    diff_pooler = np.max(np.abs(hf_pooler - k_pooler))
    diff_logits = np.max(np.abs(hf_logits - k_logits))

    assert diff_hidden < 1e-4, f"Pretrained DNABERT hidden diff too large: {diff_hidden}"
    assert diff_pooler < 1e-4, f"Pretrained DNABERT pooler diff too large: {diff_pooler}"
    assert diff_logits < 5e-4, f"Pretrained DNABERT logits diff too large: {diff_logits}"

