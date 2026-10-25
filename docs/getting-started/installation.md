# Installation

## Installing Keras-Bio

Install the latest release from PyPI:

```bash
pip install keras-bio
```

### Optional Extras

To install dependencies for PyTorch or Hugging Face weight conversion:

```bash
# PyTorch support
pip install "keras-bio[torch]"

# Hugging Face transformers weight porting utilities
pip install "keras-bio[transformers]"
```

### Development Install

To work on `keras-bio` itself, clone the repository and install in editable mode:

```bash
git clone https://github.com/anas-rz/keras-bio.git
cd keras-bio

# Documentation build dependencies
pip install -e ".[docs]"

# Full development suite
pip install -e ".[dev]"
```

---

## Selecting Your Backend

`keras-bio` operates on top of **Keras 3**. You can choose your backend before running any Python code by setting the `KERAS_BACKEND` environment variable.

=== "JAX"
    ```bash
    export KERAS_BACKEND=jax
    python your_script.py
    ```

=== "PyTorch"
    ```bash
    export KERAS_BACKEND=torch
    python your_script.py
    ```

=== "TensorFlow"
    ```bash
    export KERAS_BACKEND=tensorflow
    python your_script.py
    ```

Alternatively, set it programmatically before importing `keras`:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "torch", "tensorflow"

import keras
import keras_bio
print("Active backend:", keras.backend.backend())
```

