"""Self-contained protein sequence tokenizer for ProtT5.

Does not require external C++ libraries (sentencepiece/tiktoken/transformers)
while reproducing exact token IDs and behavior of Rostlab's ProtT5 tokenizer.
"""

import re
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Amino acid token mapping matching Rostlab/prot_t5_xl_half_uniref50-enc
_AA_TO_ID = {
    "A": 3,
    "L": 4,
    "G": 5,
    "V": 6,
    "S": 7,
    "R": 8,
    "E": 9,
    "D": 10,
    "T": 11,
    "I": 12,
    "P": 13,
    "K": 14,
    "F": 15,
    "Q": 16,
    "N": 17,
    "Y": 18,
    "M": 19,
    "H": 20,
    "W": 21,
    "C": 22,
    "X": 23,
    "B": 24,
    "O": 25,
    "U": 26,
    "Z": 27,
}

_ID_TO_AA = {v: k for k, v in _AA_TO_ID.items()}
_ID_TO_AA[0] = "<pad>"
_ID_TO_AA[1] = "</s>"
_ID_TO_AA[2] = "<unk>"
for i in range(100):
    _ID_TO_AA[28 + i] = f"<extra_id_{99 - i}>"


def clean_protein_sequence(sequence: str, replace_rare: bool = True) -> str:
    """Clean and normalize protein sequence string.

    Removes spaces/newlines and replaces rare/ambiguous residues (U, Z, O, B) with X.
    """
    cleaned = re.sub(r"\s+", "", sequence.upper())
    if replace_rare:
        cleaned = re.sub(r"[UZOB]", "X", cleaned)
    return cleaned


def format_for_t5(sequence: str) -> str:
    """Format protein sequence with spaces between residues for SentencePiece."""
    return " ".join(list(sequence))


class ProtT5Tokenizer:
    """Dependency-free protein tokenizer for ProtT5 models.

    Matches Rostlab's ProtT5 vocabulary and tokenization conventions:
    - Special tokens: <pad>=0, </s>=1, <unk>=2
    - Standard amino acids: IDs 3 to 23
    - Rare amino acids (U, Z, O, B) mapped to X (ID 23) by default
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        unk_token_id: int = 2,
        replace_rare: bool = True,
    ):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.replace_rare = replace_rare
        self.vocab_size = 128

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a single protein sequence string into list of token IDs."""
        cleaned = clean_protein_sequence(sequence, replace_rare=self.replace_rare)
        token_ids = [_AA_TO_ID.get(aa, self.unk_token_id) for aa in cleaned]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs into a protein sequence string."""
        residues = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.pad_token_id, self.eos_token_id, self.unk_token_id):
                continue
            if tid in _ID_TO_AA:
                tok = _ID_TO_AA[tid]
                if not tok.startswith("<"):
                    residues.append(tok)
            else:
                if not skip_special_tokens:
                    residues.append("<unk>")
        return "".join(residues)

    def __call__(
        self,
        sequences: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "np",
    ) -> Dict[str, Any]:
        """Tokenize a sequence or batch of sequences.

        Args:
            sequences: String or list of protein sequence strings.
            add_special_tokens: Whether to append EOS token (</s>).
            padding: True or "longest" to pad batch to maximum sequence length.
            max_length: Optional maximum sequence truncation length.
            return_tensors: "np" for numpy arrays, "keras" for Keras tensors, None for lists.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        all_ids = []
        for seq in sequences:
            ids = self.encode(seq, add_special_tokens=add_special_tokens)
            if max_length is not None:
                ids = ids[:max_length]
            all_ids.append(ids)

        if padding:
            max_len = max(len(ids) for ids in all_ids) if all_ids else 0
            if max_length is not None:
                max_len = min(max_len, max_length)

            padded_ids = []
            attention_masks = []
            for ids in all_ids:
                cur_len = len(ids)
                pad_len = max_len - cur_len
                padded = ids + [self.pad_token_id] * pad_len
                mask = [1] * cur_len + [0] * pad_len
                padded_ids.append(padded)
                attention_masks.append(mask)
            all_ids = padded_ids
            all_masks = attention_masks
        else:
            all_masks = [[1] * len(ids) for ids in all_ids]

        if return_tensors == "np":
            return {
                "input_ids": np.array(all_ids, dtype=np.int32),
                "attention_mask": np.array(all_masks, dtype=np.int32),
            }
        elif return_tensors in ("keras", "tensor"):
            import keras.ops as ops
            return {
                "input_ids": ops.convert_to_tensor(all_ids, dtype="int32"),
                "attention_mask": ops.convert_to_tensor(all_masks, dtype="int32"),
            }
        return {"input_ids": all_ids, "attention_mask": all_masks}
