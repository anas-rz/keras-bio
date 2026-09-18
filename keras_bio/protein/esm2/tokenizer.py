"""Zero-dependency protein sequence tokenizer for ESM-2.

Reproduces the exact token vocabulary and encoding behavior of ESM-2
without requiring Hugging Face transformers or any external C/Rust libraries.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np


ESM_VOCAB: List[str] = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>",
]


class ESMTokenizer:
    """Protein sequence tokenizer matching ESM-2 token IDs.

    Special token mappings:
        <cls>: 0
        <pad>: 1
        <eos>: 2
        <unk>: 3
        <mask>: 32
    """

    def __init__(self):
        self.vocab: List[str] = list(ESM_VOCAB)
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.vocab)}

        self.cls_token: str = "<cls>"
        self.pad_token: str = "<pad>"
        self.eos_token: str = "<eos>"
        self.unk_token: str = "<unk>"
        self.mask_token: str = "<mask>"

        self.cls_token_id: int = 0
        self.pad_token_id: int = 1
        self.eos_token_id: int = 2
        self.unk_token_id: int = 3
        self.mask_token_id: int = 32

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def _normalize_seq(self, seq: str) -> str:
        """Removes whitespace and capitalizes amino acid letters."""
        return "".join(seq.split()).upper()

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
        """Encodes a single protein sequence string into a list of token IDs.

        Args:
            seq: Amino acid string (e.g. "MKTVRQER").
            add_special_tokens: Whether to prepend <cls> and append <eos>.
            max_length: Maximum allowed length.
            truncation: Whether to truncate sequences longer than max_length.

        Returns:
            List of integer token IDs.
        """
        chars = self.tokenize(seq)
        token_ids = [self.token_to_id.get(ch, self.unk_token_id) for ch in chars]

        if add_special_tokens:
            if truncation and max_length is not None and len(token_ids) > max_length - 2:
                token_ids = token_ids[: max_length - 2]
            token_ids = [self.cls_token_id] + token_ids + [self.eos_token_id]
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
        """Batch-encodes multiple protein sequences with optional padding.

        Args:
            sequences: List of amino acid sequence strings.
            add_special_tokens: Whether to prepend <cls> and append <eos>.
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
        """Callable interface matching Hugging Face tokenizer convention."""
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
        """Decodes token IDs back into an amino acid sequence string."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        tokens: List[str] = []
        special_ids = {
            self.cls_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.mask_token_id,
        }
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tok = self.id_to_token.get(tid, self.unk_token)
            tokens.append(tok)

        return "".join(tokens)

