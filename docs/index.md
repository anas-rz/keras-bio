<div class="hero-container">
  <div class="hero-title">Keras-Bio</div>
  <div class="hero-subtitle">
    Multi-Backend Biological Sequence & 3D Structure Foundation Models in Keras 3.
    Run seamlessly on <strong>TensorFlow</strong>, <strong>PyTorch</strong>, and <strong>JAX</strong>.
  </div>
  <div class="hero-buttons">
    <a href="getting-started/installation/" class="btn-primary">Get Started</a>
    <a href="models/structure/esmfold/" class="btn-secondary">Explore ESMFold</a>
    <a href="https://github.com/anas-rz/keras-bio" class="btn-secondary">GitHub</a>
  </div>
</div>

<div class="grid-cards">
  <div class="card">
    <h3><span>🧬</span> DNA & Genomics</h3>
    <p>Subquadratic long-context convolution with <strong>HyenaDNA</strong> up to 1M bp and multi-scale k-mer representations with <strong>DNABERT</strong>.</p>
  </div>
  <div class="card">
    <h3><span>🧪</span> Protein Sequences</h3>
    <p>State-of-the-art representation learning with <strong>ESM-2</strong> (RoPE + APC contacts) and <strong>ProtT5</strong> encoder-decoder models.</p>
  </div>
  <div class="card">
    <h3><span>📐</span> 3D Protein Folding</h3>
    <p>Direct sequence-to-3D atomic coordinate prediction with <strong>ESMFold</strong>, Invariant Point Attention (IPA), and PDB file generation.</p>
  </div>
  <div class="card">
    <h3><span>⚡</span> Tri-Backend Parity</h3>
    <p>Native multi-backend execution on <strong>JAX</strong> (TPU/XLA), <strong>PyTorch</strong> (CUDA), and <strong>TensorFlow</strong> (Serving/TFLite) with zero lock-in.</p>
  </div>
</div>

---

## Foundation Models Matrix

| Category | Model Architecture | Pretrained Checkpoints | Native Backends | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Protein Sequences** | **ProtT5** | `prot_t5_xl_half_uniref50-enc`, `prot_t5_xl_uniref50` | TensorFlow, PyTorch, JAX | Ready |
| **Protein Sequences** | **ESM-2** | `esm2_t6_8M`, `esm2_t12_35M`, `esm2_t30_150M`, `esm2_t33_650M`, `esm2_t36_3B`, `esm2_t48_15B` | TensorFlow, PyTorch, JAX | Ready |
| **DNA Sequences** | **DNABERT** | `dnabert_3mer`, `dnabert_4mer`, `dnabert_5mer`, `dnabert_6mer` | TensorFlow, PyTorch, JAX | Ready |
| **DNA Sequences** | **HyenaDNA** | `hyenadna_tiny_1k`, `hyenadna_small_32k`, `hyenadna_medium_160k`, `hyenadna_large_1m` | TensorFlow, PyTorch, JAX | Ready |
| **3D Structures** | **ESMFold** | `esmfold_v1`, `esmfold_tiny` | TensorFlow, PyTorch, JAX | Ready |

---

## Rapid Demonstration

=== "3D Structure Folding"

    ```python
    from keras_bio.structure.esmfold import ESMFoldConfig, ESMFoldModel

    # Load ESMFold model
    model = ESMFoldModel(ESMFoldConfig.from_preset("esmfold_v1"))

    # Direct folding from amino acid sequences to PDB files
    result = model.infer("MLKNVQVQLV", num_recycles=3)

    print("Mean pLDDT confidence:", result["plddt"].mean())
    with open("folded_protein.pdb", "w") as f:
        f.write(result["pdbs"][0])
    ```

=== "Protein Embeddings (ESM-2)"

    ```python
    from keras_bio.protein.esm2 import ESM2Config, ESM2Model, ESMTokenizer

    tokenizer = ESMTokenizer()
    model = ESM2Model(ESM2Config.from_preset("esm2_t6_8M"))

    sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    embeddings = model.embed(sequences, tokenizer=tokenizer, pool="mean")
    print("Embedding vector shape:", embeddings.shape)  # [1, 320]
    ```

=== "Genomic Long-Context (HyenaDNA)"

    ```python
    from keras_bio.dna.hyena import HyenaDNAConfig, HyenaDNAModel, HyenaDNATokenizer

    tokenizer = HyenaDNATokenizer()
    model = HyenaDNAModel(HyenaDNAConfig.from_preset("hyenadna_tiny_1k"))

    dna_seqs = ["ACTG" * 256]
    embeddings = model.embed(dna_seqs, tokenizer=tokenizer, pool="mean")
    print("Genomic embedding shape:", embeddings.shape)  # [1, 128]
    ```
