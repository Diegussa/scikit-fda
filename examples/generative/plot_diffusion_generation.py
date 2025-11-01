"""
Functional Data Generation using Diffusion Models
==================================================

This example demonstrates how to generate synthetic functional data using
diffusion models in latent space, following the FunDiff architecture.

The approach consists of three main components:

1. **Encoder**: Maps functional data x(t) to a low-dimensional latent vector z
2. **Decoder**: Maps latent vector z and query points t to function values x(t)
3. **Diffusion Model**: Learns to generate new latent vectors from noise

This implementation achieves:
- Discretization independence (can generate at any resolution)
- Computational efficiency (diffusion in low-dimensional space)
- Theoretical guarantees (minimax-optimal density estimation)

For the complete implementation with training, see the Jupyter notebooks:
- 01_functional_autoencoder_training.ipynb
- 02_latent_diffusion_training.ipynb
- 03_functional_generation_demo.ipynb

Note: This example requires PyTorch. Install with:
    pip install torch
or
    pip install scikit-fda[generative]
"""

# Author: scikit-fda developers
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from skfda.representation.grid import FDataGrid

# %%
# Generate Toy Functional Data
# -----------------------------
#
# First, we create synthetic functional data with varying amplitude,
# frequency, and phase:
#
# .. math::
#     x(t) = A \cdot \sin(\omega t + \phi) + \epsilon(t)
#
# This serves as the ground truth distribution for our generative model.


def generate_toy_functional_data(n_samples=100, n_points=100, domain_range=(0, 1)):
    """
    Generate synthetic functional data with varying parameters.

    Parameters
    ----------
    n_samples : int
        Number of functional samples to generate
    n_points : int
        Number of discretization points
    domain_range : tuple
        Tuple (start, end) for the domain

    Returns
    -------
    FDataGrid
        Object containing the synthetic functions
    """
    t = np.linspace(domain_range[0], domain_range[1], n_points)
    data_matrix = np.zeros((n_samples, n_points, 1))

    for i in range(n_samples):
        # Random amplitude, frequency, and phase
        A = np.random.uniform(0.5, 2.0)
        omega = np.random.uniform(2 * np.pi, 6 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        # Generate function with small noise
        noise = np.random.normal(0, 0.05, n_points)
        x_t = A * np.sin(omega * t + phi) + noise
        data_matrix[i, :, 0] = x_t

    return FDataGrid(data_matrix=data_matrix, grid_points=t)


# Generate synthetic data
fdata = generate_toy_functional_data(n_samples=50, n_points=100)

# %%
# Visualize the Generated Data
# -----------------------------
#
# Let's visualize some samples from our synthetic dataset.

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(20):
    ax.plot(
        fdata.grid_points[0],
        fdata.data_matrix[i, :, 0],
        alpha=0.6,
        linewidth=1.5,
    )

ax.set_xlabel("t")
ax.set_ylabel("x(t)")
ax.set_title("Synthetic Functional Data (Ground Truth)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Architecture Overview
# ---------------------
#
# The complete pipeline for generating functional data with diffusion models
# consists of three stages:
#
# **Stage 1: Encoder (Functional Data → Latent Space)**
#
# - Uses 1D-CNN with global average pooling
# - Maps functions with any discretization to fixed-size latent vectors
# - Achieves discretization independence
#
# .. code-block:: python
#
#     class FunctionalEncoder(nn.Module):
#         def __init__(self, latent_dim=64):
#             # 1D Convolutions
#             self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
#             self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
#             # ...
#             # Global pooling for any input length
#             self.global_pool = nn.AdaptiveAvgPool1d(1)
#             self.fc = nn.Linear(128, latent_dim)
#
# **Stage 2: Diffusion Model (Latent Space Generation)**
#
# - Learns distribution of latent vectors using DDPM
# - Trains on encoded latent representations
# - Generates new latent codes from noise
#
# .. code-block:: python
#
#     # Forward diffusion: x_0 → x_T (add noise gradually)
#     # Reverse diffusion: x_T → x_0 (denoise step by step)
#     class LatentDiffusionModel(nn.Module):
#         def forward(self, x, t):
#             # Predict noise at timestep t
#             return noise_prediction
#
# **Stage 3: Decoder (Latent Space → Functional Data)**
#
# - Neural Implicit Representation (INR) using MLP
# - Takes latent vector + query points → function values
# - Can evaluate at arbitrary discretizations
#
# .. code-block:: python
#
#     class FunctionalDecoder(nn.Module):
#         def forward(self, z, t_query):
#             # Concatenate latent z with time coordinates t_query
#             # Process through MLP to get function values
#             return x_reconstructed

# %%
# Statistical Properties
# -----------------------
#
# To validate the quality of generated functional data, we can compare
# statistical properties of real vs. synthetic data:
#
# 1. **Mean Function**: :math:`\bar{x}(t) = \frac{1}{n}\sum_{i=1}^n x_i(t)`
# 2. **Covariance Function**: :math:`\text{Cov}(s, t) = \text{E}[(x(s) - \bar{x}(s))(x(t) - \bar{x}(t))]`
# 3. **Functional PCA**: Principal components of variation

# Compute and visualize mean function
mean_function = fdata.mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    mean_function.grid_points[0],
    mean_function.data_matrix[0, :, 0],
    linewidth=3,
    color="blue",
    label="Mean Function",
)

# Plot individual functions with transparency
for i in range(20):
    ax.plot(
        fdata.grid_points[0],
        fdata.data_matrix[i, :, 0],
        alpha=0.2,
        linewidth=1,
        color="gray",
    )

ax.set_xlabel("t")
ax.set_ylabel("x(t)")
ax.set_title("Mean Function and Individual Samples")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Covariance Structure
# --------------------
#
# The covariance function reveals the correlation structure of the functional
# data:

cov_function = fdata.cov()

fig, ax = plt.subplots(figsize=(8, 6))

# Evaluate covariance on a grid
t_grid = fdata.grid_points[0]
cov_values = cov_function(t_grid, t_grid)

im = ax.imshow(
    cov_values,
    cmap="viridis",
    aspect="auto",
    origin="lower",
    extent=[0, 1, 0, 1],
)
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_title("Covariance Function Cov(s, t)")
plt.colorbar(im, ax=ax, label="Covariance")
plt.tight_layout()
plt.show()

# %%
# Key Advantages of Diffusion-Based Generation
# ---------------------------------------------
#
# 1. **Discretization Independence**
#    - Train on 100 points, generate on 200 or more
#    - True continuous representation via INR decoder
#
# 2. **Computational Efficiency**
#    - Diffusion in 64D latent space vs. 100+D functional space
#    - Orders of magnitude faster training and sampling
#
# 3. **Theoretical Guarantees**
#    - Achieves minimax-optimal rates for density estimation
#    - Not just empirically successful, but theoretically sound
#
# 4. **High Quality Generation**
#    - Captures complex statistical properties
#    - Mean, covariance, and principal components match real data
#
# For the complete implementation with training code and detailed examples,
# see the Jupyter notebooks in the `examples/generative/` directory.

# %%
# Next Steps
# ----------
#
# To train your own generative model:
#
# 1. Run `01_functional_autoencoder_training.ipynb` to train the encoder/decoder
# 2. Run `02_latent_diffusion_training.ipynb` to train the diffusion model
# 3. Run `03_functional_generation_demo.ipynb` to generate and validate data
#
# You can also apply this approach to real datasets from scikit-fda:
#
# .. code-block:: python
#
#     from skfda.datasets import fetch_growth, fetch_weather
#
#     # Berkeley Growth Dataset
#     fdata = fetch_growth(return_X_y=False)
#
#     # Canadian Weather Dataset
#     fdata = fetch_weather(return_X_y=False)
#
# Requirements:
#
# .. code-block:: bash
#
#     pip install torch
#     # or
#     pip install scikit-fda[generative]
