# Multi-Backend Guide

Keras 3 provides a hardware-agnostic tensor execution engine. Every model, layer, and mathematical operation in `keras-bio` is built on top of `keras.ops` and standard Keras layer primitives.

## Why Multi-Backend?

1. **JAX for Research & Scaling**: Take advantage of automatic compilation with XLA, TPUs, and just-in-time compilation.
2. **PyTorch for Ecosystem Integration**: Easily connect `keras-bio` models with existing PyTorch data loaders, training loops, or Hugging Face toolkits.
3. **TensorFlow for Production Serving**: Export directly to TensorFlow SavedModel, TFLite for embedded and edge inference, or TensorFlow Serving.

---

## Best Practices in Keras-Bio

### 1. Converting Tensors to NumPy

Never call `.numpy()` directly on a tensor, as this method is backend-specific and may fail on JAX or PyTorch depending on device placement. Instead, use:

```python
from keras import ops

# Recommended: works universally across JAX, Torch, and TensorFlow
array_np = ops.convert_to_numpy(tensor)
```

### 2. Creating Tensors

```python
import numpy as np
from keras import ops

x_np = np.zeros((2, 10), dtype="float32")
x_tensor = ops.convert_to_tensor(x_np)
```

### 3. Model Saving & Serialization

Save models using the native `.keras` zip archive format:

```python
# Save model
model.save("my_protein_model.keras")

# Load model on any backend
import keras
loaded_model = keras.models.load_model("my_protein_model.keras")
```

