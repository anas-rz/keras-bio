"""Exact numerical parity verification for HyenaDNA against Hugging Face PyTorch."""

import keras.ops as ops
import numpy as np
import pytest
import torch

from keras_bio.dna.hyena.config import HyenaDNAConfig
from keras_bio.dna.hyena.convert import convert_hyenadna_weights_from_hf
from keras_bio.dna.hyena.modeling import HyenaDNAForCausalLM
from keras_bio.dna.hyena.tokenizer import HyenaDNATokenizer


def test_hyenadna_pretrained_checkpoint_parity():
    """Verify parity against official LongSafari/hyenadna-tiny-1k-seqlen-hf checkpoint."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo_id = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
        pt_model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True).eval()
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load checkpoint: {e}")

    cfg = HyenaDNAConfig.from_preset("hyenadna_tiny_1k")
    k_model = HyenaDNAForCausalLM(cfg)
    k_model.build((None, 16))

    convert_hyenadna_weights_from_hf(pt_model.state_dict(), k_model)

    seq = "ATCGATCGATCG"
    inputs = tok(seq, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()

    with torch.no_grad():
        pt_logits = pt_model(**inputs).logits.numpy()

    k_out = k_model(input_ids)
    k_logits = ops.convert_to_numpy(k_out["logits"])

    diff = np.max(np.abs(pt_logits - k_logits))
    assert diff < 1e-4, f"Pretrained HyenaDNA logits parity diff too large: {diff}"

