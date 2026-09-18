"""Zero-dependency k-mer DNA tokenizer for DNABERT models.

Reproduces the exact token vocabulary and encoding behavior of DNABERT (3, 4, 5, 6-mer)
without requiring Hugging Face transformers or any external C/Rust tokenizers.
"""

import itertools
from typing import Any, Dict, List, Optional, Union

import numpy as np


class DNABERTTokenizer:
    """k-mer DNA sequence tokenizer matching official DNABERT checkpoints.

    Supports automatic sliding window k-merization from raw DNA sequences
    as well as pre-tokenized space-separated k-mer strings.

    Vocabulary structure:
        0: [PAD]
        1: [UNK]
        2: [CLS]
        3: [SEP]
        4: [MASK]
        Followed by all 4^k k-mers generated from bases ['A', 'T', 'C', 'G'].
    """

    def __init__(self, kmer: int = 6):
        """Initializes the tokenizer with the specified k-mer size.

        Args:
            kmer: Length of k-mers (typically 3, 4, 5, or 6). Defaults to 6.
        """
        if kmer not in (3, 4, 5, 6):
            raise ValueError(f"kmer must be 3, 4, 5, or 6, got {kmer}")
        self.kmer = kmer

        self.pad_token: str = "[PAD]"
        self.unk_token: str = "[UNK]"
        self.cls_token: str = "[CLS]"
        self.sep_token: str = "[SEP]"
        self.mask_token: str = "[MASK]"

        self.pad_token_id: int = 0
        self.unk_token_id: int = 1
        self.cls_token_id: int = 2
        self.sep_token_id: int = 3
        self.mask_token_id: int = 4

        bases = ["A", "T", "C", "G"]
        kmer_tokens = ["".join(p) for p in itertools.product(bases, repeat=kmer)]
        self.vocab: List[str] = [
            self.pad_token,
            self.unk_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
        ] + kmer_tokens

        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def _normalize_seq(self, seq: str) -> str:
        """Removes whitespace, capitalizes nucleotides, and converts RNA 'U' to 'T'."""
        return "".join(seq.split()).upper().replace("U", "T")

    def dna_to_kmers(self, seq: str, k: Optional[int] = None) -> List[str]:
        """Converts a continuous DNA sequence into sliding k-mers.

        Args:
            seq: Raw continuous DNA string (e.g. "ATCGATCG").
            k: k-mer length. If None, uses tokenizer's kmer.

        Returns:
            List of k-mer strings.
        """
        k = k or self.kmer
        norm_seq = self._normalize_seq(seq)
        if len(norm_seq) < k:
            return [norm_seq] if norm_seq else []
        return [norm_seq[i : i + k] for i in range(len(norm_seq) - k + 1)]

    def kmers_to_dna(self, kmers: List[str]) -> str:
        """Reconstructs continuous DNA sequence from sliding k-mers."""
        if not kmers:
            return ""
        dna = [kmers[0]]
        for kmer in kmers[1:]:
            dna.append(kmer[-1])
        return "".join(dna)

    def tokenize(self, seq: str) -> List[str]:
        """Tokenizes sequence string into k-mer tokens.

        If `seq` contains whitespace, splits by whitespace (space-separated k-mers).
        Otherwise, converts continuous DNA sequence into sliding k-mers.
        """
        stripped = seq.strip()
        if " " in stripped:
            return stripped.upper().split()
        return self.dna_to_kmers(stripped)

    def encode(
        self,
        seq: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """Encodes a single DNA sequence or k-mer string into token IDs.

        Args:
            seq: Continuous DNA string or space-separated k-mers.
            add_special_tokens: Whether to prepend [CLS] and append [SEP].
            max_length: Maximum allowed sequence length in tokens.
            truncation: Whether to truncate sequences longer than max_length.

        Returns:
            List of integer token IDs.
        """
        tokens = self.tokenize(seq)
        token_ids = [self.token_to_id.get(tok, self.unk_token_id) for tok in tokens]

        if add_special_tokens:
            if truncation and max_length is not None and len(token_ids) > max_length - 2:
                token_ids = token_ids[: max_length - 2]
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
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
            sequences: List of continuous DNA strings or space-separated k-mer strings.
            add_special_tokens: Whether to prepend [CLS] and append [SEP].
            padding: Whether to pad sequences to the longest sequence in the batch
                (True or "longest") or to max_length ("max_length").
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences longer than max_length.
            return_tensors: "np" for numpy arrays, "keras" for Keras tensors,
                or None for nested Python lists.

        Returns:
            Dictionary with "input_ids", "attention_mask", and "token_type_ids".
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
            token_type_ids: List[List[int]] = []

            for ids in encoded_list:
                pad_count = max(0, target_len - len(ids))
                padded = ids + [self.pad_token_id] * pad_count
                mask = [1] * len(ids) + [0] * pad_count
                types = [0] * target_len
                padded_ids.append(padded)
                attention_masks.append(mask)
                token_type_ids.append(types)

            if return_tensors == "np":
                return {
                    "input_ids": np.array(padded_ids, dtype=np.int32),
                    "attention_mask": np.array(attention_masks, dtype=np.int32),
                    "token_type_ids": np.array(token_type_ids, dtype=np.int32),
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
                    "token_type_ids": keras.ops.convert_to_tensor(
                        np.array(token_type_ids, dtype=np.int32)
                    ),
                }
            return {
                "input_ids": padded_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids,
            }

        # No padding
        if return_tensors == "np":
            return {
                "input_ids": np.array(encoded_list, dtype=object),
                "attention_mask": np.array([[1] * len(ids) for ids in encoded_list], dtype=object),
                "token_type_ids": np.array([[0] * len(ids) for ids in encoded_list], dtype=object),
            }
        return {
            "input_ids": encoded_list,
            "attention_mask": [[1] * len(ids) for ids in encoded_list],
            "token_type_ids": [[0] * len(ids) for ids in encoded_list],
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
        """Callable interface matching Hugging Face tokenizer conventions."""
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
        """Decodes token IDs back into space-separated k-mers."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        tokens: List[str] = []
        special_ids = {
            self.cls_token_id,
            self.pad_token_id,
            self.sep_token_id,
            self.mask_token_id,
        }
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tok = self.id_to_token.get(tid, self.unk_token)
            tokens.append(tok)

        return " ".join(tokens)

