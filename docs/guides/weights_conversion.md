# Weight Porting from PyTorch & Hugging Face

`keras-bio` provides weight porting converters that copy weights from official Hugging Face PyTorch state dicts into Keras 3 models.

## Conversion Principles

- **Dense Layer Transposition**: PyTorch stores linear layer weights as `[out_features, in_features]`, whereas Keras stores kernel matrices as `[in_features, out_features]`. `keras-bio` converters transpose 2D kernels automatically.
- **Normalization Parameters**: PyTorch `weight` and `bias` are mapped to Keras `gamma` and `beta`.
- **Rotary & Positional Buffers**: Embeddings and non-trainable frequency buffers are preserved.

---

## Porting ESM-2 Weights

```python
import torch
from transformers import EsmForMaskedLM as HFEsmModel

from keras_bio.protein.esm2 import ESM2Config, ESM2ForMaskedLM
from keras_bio.protein.esm2.convert import convert_esm2_weights_from_hf

# 1. Load Hugging Face PyTorch model
pt_model = HFEsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# 2. Instantiate Keras model
k_config = ESM2Config.from_hf(pt_model.config.to_dict())
k_model = ESM2ForMaskedLM(k_config)
k_model.build((None, 128))

# 3. Port weights into Keras
convert_esm2_weights_from_hf(pt_model.state_dict(), k_model)

# 4. Save to .keras archive
k_model.save("esm2_t6_8m.keras")
```

---

## Porting ESMFold Weights

```python
from transformers import AutoModelForProteinFolding

from keras_bio.structure.esmfold import ESMFoldConfig, ESMFoldModel
from keras_bio.structure.esmfold.convert import convert_esmfold_weights_from_hf

# 1. Load official ESMFold checkpoint
hf_model = AutoModelForProteinFolding.from_pretrained("facebook/esmfold_v1")

# 2. Instantiate Keras ESMFoldModel
config = ESMFoldConfig.from_hf(hf_model.config.to_dict())
k_model = ESMFoldModel(config)

# 3. Transfer weights
convert_esmfold_weights_from_hf(hf_model.state_dict(), k_model)
k_model.save("esmfold_v1.keras")
```

