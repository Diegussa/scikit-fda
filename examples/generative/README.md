# Generative Models for Functional Data

This directory contains a complete implementation of **functional data generation using latent space diffusion models**, following the FunDiff architecture.

## Overview

The implementation demonstrates how to generate synthetic functional data using state-of-the-art diffusion models. The approach is based on:

1. **Functional Autoencoder**: Encodes functions into a low-dimensional latent space
2. **Latent Diffusion**: Learns the distribution of latent codes using diffusion models
3. **Neural Implicit Representation (INR)**: Decodes latent codes to functions at arbitrary discretizations

## Architecture

```
Functional Data x(t)  →  [Encoder]  →  Latent Code z ∈ R^d
                                             ↓
                                      [Diffusion Model]
                                             ↓
                                    Sample new z' from noise
                                             ↓
Synthetic x'(t)  ←  [Decoder(z', t)]  ←  Query points t
```

## Key Features

- **Discretization Independence**: Generate functions at any resolution (train on 100 points, generate on 200)
- **Computational Efficiency**: Diffusion in low-dimensional latent space (e.g., 64D) instead of high-dimensional functional space
- **Theoretical Guarantees**: Based on minimax-optimal density estimation for functional data
- **scikit-fda Integration**: Full compatibility with FDataGrid and other scikit-fda classes

## Notebooks

### 1. `01_functional_autoencoder_training.ipynb`
**Training the Functional Autoencoder**

- Implements the **Encoder** (1D-CNN with global pooling)
- Implements the **Decoder** (Neural Implicit Representation using MLP)
- Trains on synthetic functional data
- Demonstrates discretization independence
- Saves trained models for Phase 2

**Key concepts**:
- Encoder: Maps `x(t)` with any discretization → fixed-size latent `z`
- Decoder: Maps `z` + query points `t_q` → function values `x(t_q)`

### 2. `02_latent_diffusion_training.ipynb`
**Training the Diffusion Model in Latent Space**

- Loads pre-trained Encoder
- Encodes all training data into latent vectors
- Implements **DDPM (Denoising Diffusion Probabilistic Model)**
- Trains diffusion model in low-dimensional latent space
- Tests sampling from the diffusion model

**Key concepts**:
- Forward process: Gradually add Gaussian noise to latent codes
- Reverse process: Learn to denoise and generate new latent codes
- Efficiency: Training in 64D instead of 100+D

### 3. `03_functional_generation_demo.ipynb`
**Complete Generation Pipeline**

- Loads all pre-trained models
- Demonstrates the full generation workflow:
  1. Sample latent code from diffusion (noise → clean latent)
  2. Decode to functional data at any discretization
  3. Create FDataGrid objects
- Statistical validation:
  - Visual comparison
  - Mean function comparison
  - Covariance structure comparison
  - Functional PCA analysis

## Installation Requirements

In addition to scikit-fda, you'll need PyTorch for running these notebooks:

```bash
pip install torch torchvision
```

Or with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

Run the notebooks in order:

```bash
# 1. Train the autoencoder
jupyter notebook 01_functional_autoencoder_training.ipynb

# 2. Train the diffusion model
jupyter notebook 02_latent_diffusion_training.ipynb

# 3. Generate and validate synthetic data
jupyter notebook 03_functional_generation_demo.ipynb
```

## Theoretical Background

This implementation is based on the paper:

**FunDiff: Diffusion Models for Functional Data**

Key theoretical results:
- Diffusion models achieve **minimax-optimal rates** for density estimation in infinite-dimensional spaces
- The approach is not just empirically successful but theoretically sound for FDA
- Provides guarantees under mild smoothness assumptions

## Extending the Implementation

### Use Real Datasets

Replace the toy data generator with scikit-fda datasets:

```python
from skfda.datasets import fetch_growth, fetch_weather

# Berkeley Growth Dataset
fdata = fetch_growth(return_X_y=False)

# Canadian Weather Dataset
fdata = fetch_weather(return_X_y=False)
```

### Conditional Generation

Add conditional information to the diffusion model for controlled generation (e.g., generate functions with specific properties).

### Advanced Validation

Implement **TSTR (Train on Synthetic, Test on Real)**:
1. Train a classifier on synthetic data
2. Test on real data
3. High accuracy indicates good generation quality

### Alternative Architectures

- **Transformers**: Replace 1D-CNN encoder with Transformer
- **SIREN**: Use periodic activations in the decoder for smoother functions
- **Score-Based SDEs**: Implement continuous-time diffusion

## Citation

If you use this implementation in your research, please cite both scikit-fda and the relevant diffusion model papers.

## License

This code follows the same license as scikit-fda (BSD 3-Clause).

## Contact

For questions or issues, please open an issue on the scikit-fda GitHub repository.
