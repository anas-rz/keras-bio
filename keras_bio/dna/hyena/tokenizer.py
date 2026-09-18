"""Zero-dependency character-level nucleotide tokenizer for HyenaDNA models.

Reproduces the exact token vocabulary and encoding behavior of official HyenaDNA checkpoints
without requiring external C/Rust tokenizers or Hugging Face transformers.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np


HYENADNA_VOCAB: Dict[str, int] = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[BOS]": 2,
    "[MASK]": 3,
    "[PAD]": 4,
    "[RESERVED]": 5,
    "[UNK]": 6,
    "A": 7,
    "C": 8,
    "G": 9,
    "T": 10,
    "N": 11,
}


class HyenaDNATokenizer:
    """Character-level nucleotide tokenizer matching official HyenaDNA checkpoints.

    Token mapping:
        [CLS]: 0, [SEP]: 1, [BOS]: 2, [MASK]: 3, [PAD]: 4, [RESERVED]: 5, [UNK]: 6,
        A: 7, C: 8, G: 9, T: 10, N: 11.
    """

    def __init__(self, max_length: int = 1026):
        self.max_length = max_length
        self.token_to_id: Dict[str, int] = dict(HYENADNA_VOCAB)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        self.cls_token: str = "[CLS]"
        self.sep_token: str = "[SEP]"
        self.bos_token: str = "[BOS]"
        self.mask_token: str = "[MASK]"
        self.pad_token: str = "[PAD]"
        self.unk_token: str = "[UNK]"

        self.cls_token_id: int = 0
        self.sep_token_id: int = 1
        self.bos_token_id: int = 2
        self.mask_token_id: int = 3
        self.pad_token_id: int = 4
        self.unk_token_id: int = 6

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def _normalize_seq(self, seq: str) -> str:
        """Removes whitespace, capitalizes nucleotides, and converts RNA 'U' to 'T'."""
        return "".join(seq.split()).upper().replace("U", "T")

    def tokenize(self, seq: str) -> List[str]:
        """Splits normalized sequence into character tokens."""
        norm = self._normalize_seq(seq)
        return list(norm)

    def encode(
        self,
        seq: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """Encodes a single DNA sequence string into token IDs.

        Args:
            seq: Continuous DNA string (e.g. "ATCGATCG").
            add_special_tokens: Whether to append [SEP] at the end (matching HyenaDNA).
            max_length: Maximum allowed sequence length in tokens.
            truncation: Whether to truncate sequences longer than max_length.

        Returns:
            List of integer token IDs.
        """
        chars = self.tokenize(seq)
        token_ids = [self.token_to_id.get(ch, self.unk_token_id) for ch in chars]

        if add_special_tokens:
            if truncation and max_length is not None and len(token_ids) > max_length - 1:
                token_ids = token_ids[: max_length - 1]
            token_ids = token_ids + [self.sep_token_id]
        elif truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def batch_encode(
        self,
        sequences: List[str],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch-encodes multiple DNA sequences with optional padding.

        Args:
            sequences: List of continuous DNA strings.
            add_special_tokens: Whether to append [SEP].
            padding: Whether to pad sequences to the longest sequence in the batch
                (True or "longest") or to max_length ("max_length").
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences longer than max_length.
            return_tensors: "np" for numpy arrays, "keras" for Keras tensors,
                or None for nested Python lists.

        Returns:
            Dictionary with "input_ids" and "attention_mask".
        """
        encoded_list = [
            self.encode(
                s,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
            )
            for s in sequences
        ]

        if padding:
            if padding == "max_length" and max_length is not None:
                target_len = max_length
            else:
                target_len = max(len(ids) for ids in encoded_list) if encoded_list else 0

            padded_ids: List[List[int]] = []
            attention_masks: List[List[int]] = []

            for ids in encoded_list:
                pad_count = max(0, target_len - len(ids))
                padded = ids + [self.pad_token_id] * pad_count
                mask = [1] * len(ids) + [0] * pad_count
                padded_ids.append(padded)
                attention_masks.append(mask)

            if return_tensors == "np":
                return {
                    "input_ids": np.array(padded_ids, dtype=np.int32),
                    "attention_mask": np.array(attention_masks, dtype=np.int32),
                }
            elif return_tensors in ("keras", "tf", "torch", "jax"):
                import keras

                return {
                    "input_ids": keras.ops.convert_to_tensor(
                        np.array(padded_ids, dtype=np.int32)
                    ),
                    "attention_mask": keras.ops.convert_to_tensor(
                        np.array(attention_masks, dtype=np.int32)
                    ),
                }
            return {"input_ids": padded_ids, "attention_mask": attention_masks}

        # No padding
        if return_tensors == "np":
            return {
                "input_ids": np.array(encoded_list, dtype=object),
                "attention_mask": np.array([[1] * len(ids) for ids in encoded_list], dtype=object),
            }
        return {
            "input_ids": encoded_list,
            "attention_mask": [[1] * len(ids) for ids in encoded_list],
        }

    def __call__(
        self,
        sequences: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Callable interface matching standard tokenizer convention."""
        if isinstance(sequences, str):
            sequences = [sequences]
        return self.batch_encode(
            sequences=sequences,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def decode(
        self,
        token_ids: Union[List[int], np.ndarray, Any],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes token IDs back into a DNA sequence string."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        tokens: List[str] = []
        special_ids = {
            self.cls_token_id,
            self.sep_token_id,
            self.bos_token_id,
            self.mask_token_id,
            self.pad_token_id,
            self.unk_token_id,
            5,  # [RESERVED]
        }
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tok = self.id_to_token.get(tid, self.unk_token)
            tokens.append(tok)

        return "".join(tokens)

