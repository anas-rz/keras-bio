# Keras-Bio

<p align="center">
  <strong>Multi-Backend Biological Sequence & Structure Foundation Models in Keras 3</strong>
</p>

<p align="center">
  <a href="https://github.com/anas-rz/keras-bio/actions"><img src="https://img.shields.io/badge/CI-Multi--Backend-brightgreen" alt="CI"></a>
  <a href="https://keras.io"><img src="https://img.shields.io/badge/Keras-3-red.svg" alt="Keras 3"></a>
  <img src="https://img.shields.io/badge/Backends-TensorFlow%20%7C%20PyTorch%20%7C%20JAX-blue" alt="Backends">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License"></a>
</p>

---

## Welcome to Keras-Bio

`keras-bio` brings modern biological foundation models (proteins, genomics, 3D structures, and biomolecules) into the unified, production-grade **Keras 3** ecosystem. 

Write your model architecture once using standard `keras.ops` and run seamlessly across:

- **TensorFlow** (CPU, GPU, TPU, TF Serving, TFLite)
- **PyTorch** (CUDA, MPS, PyTorch ecosystem)
- **JAX** (XLA compilation, TPU, vectorized maps)

---

## Implemented Models

| Domain | Architecture | Status | Pretrained Presets | Multi-Backend |
| :--- | :--- | :---: | :--- | :---: |
| **Protein Sequences** | **ProtT5** (Encoder & Enc-Dec) | Ready | `prot_t5_xl_half_uniref50-enc`, `prot_t5_xl_uniref50` | TensorFlow, PyTorch, JAX |
| **Protein Sequences** | **ESM-2** (Evolutionary Scale Modeling) | Ready | `esm2_t6_8M`, `esm2_t12_35M`, `esm2_t30_150M`, `esm2_t33_650M`, `esm2_t36_3B`, `esm2_t48_15B` | TensorFlow, PyTorch, JAX |
| **DNA Sequences** | **DNABERT** (Genomic Language Modeling) | Ready | `dnabert_3mer`, `dnabert_4mer`, `dnabert_5mer`, `dnabert_6mer` | TensorFlow, PyTorch, JAX |
| **DNA Sequences** | **HyenaDNA** (Long-Context Subquadratic Convolution) | Ready | `hyenadna_tiny`, `hyenadna_small`, `hyenadna_medium`, `hyenadna_large` | TensorFlow, PyTorch, JAX |
| **3D Structures** | **ESMFold** (Direct Sequence-to-3D Atomic Folding) | Ready | `esmfold_v1`, `esmfold_tiny` | TensorFlow, PyTorch, JAX |

---

## Key Features

- **Pure Keras 3 Multi-Backend**: No hard backend dependencies. Choose `tensorflow`, `torch`, or `jax` by simply setting `KERAS_BACKEND`.
- **Zero External Tokenizer Dependencies**: Built-in, high-performance pure Python/NumPy tokenizers matching original vocabularies exactly.
- **Exact Numerical Parity**: Verified against Hugging Face PyTorch checkpoints to within numerical precision thresholds.
- **Production Ready**: Full serialization with standard `.keras` zip format via `model.save()` and `keras.models.load_model()`.
- **Full Structure Pipeline**: Invariant Point Attention (IPA), $SE(3)$ / $SO(3)$ geometry transformations, and standard `.pdb` file export.

