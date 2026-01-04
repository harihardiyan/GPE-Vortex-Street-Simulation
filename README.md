

# High-Performance GPE Simulation: Superfluid Flow and Vortex Shedding in 2D Channels

[![JAX](https://img.shields.io/badge/Accelerated-JAX-orange.svg)](https://github.com/google/jax)
[![Physics](https://img.shields.io/badge/Physics-Bose--Einstein%20Condensates-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains a high-performance numerical solver for the **2D Gross-Pitaevskii Equation (GPE)**, optimized for modern hardware accelerators (GPUs and TPUs) using the **JAX** framework. 

The simulation models the dynamics of a superfluid (such as a Bose-Einstein Condensate) flowing through a channel and encountering a bluff obstacle. It captures complex hydrodynamics, including the transition from laminar flow to the formation of a **Von Kármán vortex street** in a quantum fluid.

## Physical Model

The dynamics of the condensate wavefunction $\psi(\mathbf{r}, t)$ are governed by the time-dependent Gross-Pitaevskii Equation:

$$i \hbar \frac{\partial \psi}{\partial t} = \left( -\frac{\hbar^2}{2m}\nabla^2 + V_{ext}(\mathbf{r}) + g|\psi(\mathbf{r}, t)|^2 \right) \psi(\mathbf{r}, t)$$

Where:
- $-\frac{\hbar^2}{2m}\nabla^2$ is the kinetic energy operator.
- $V_{ext}(\mathbf{r})$ represents the external potential (walls + obstacle).
- $g$ is the interaction constant ($g > 0$ for repulsive interactions).
- $|\psi|^2$ is the local superfluid density.

### Potential Landscape
The total potential $V_{ext}$ is defined as:
1. **Gaussian Obstacle:** $V_{obs} = A \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right)$
2. **Channel Walls:** Rigid potential barriers at $y = \pm L_y/2$.

## Numerical Methodology

### 1. Split-Step Fourier Method (Strang Splitting)
To evolve the system in time, we use a second-order **Strang Splitting** scheme, which separates the linear and non-linear operators:
$$e^{-iH\Delta t} \approx e^{-iV \frac{\Delta t}{2}} e^{-iT \Delta t} e^{-iV \frac{\Delta t}{2}}$$
This method ensures $O(\Delta t^2)$ accuracy and preserves the unitarity (norm) of the wavefunction.

### 2. Spectral Gradient Computation
Spatial derivatives are calculated in the Fourier domain to achieve spectral accuracy:
$$\nabla \psi = \mathcal{F}^{-1} \left( i\mathbf{k} \cdot \mathcal{F}(\psi) \right)$$
This approach is significantly more robust than finite-difference methods for periodic and quasi-periodic systems.

### 3. JAX Acceleration
The implementation leverages **XLA (Accelerated Linear Algebra)** via JAX to:
- **JIT Compile** the time-stepping kernels for hardware-specific optimization.
- Support **Vectorized Sweeps** across different physical parameters (e.g., varying flow velocity $k_{0x}$).
- Utilize **float32/complex64** precision for optimal performance on Google TPUs and NVIDIA GPUs.

## Key Features & Analysis

- **Topological Vortex Counting:** Implements a robust phase-winding algorithm to detect quantized vortices by calculating the circulation $\oint \nabla S \cdot d\ell$ around grid plaquettes.
- **Drag Force Analysis:** Calculates the analytical drag force $F_x$ exerted on the obstacle by integrating the density-potential gradient product:
  $$F_x = -\int |\psi|^2 \frac{\partial V_{obs}}{\partial x} d^2\mathbf{r}$$
- **Strouhal Number Characterization:** Performs Fast Fourier Transform (FFT) on the drag force history to extract the shedding frequency $f$ and calculate the dimensionless Strouhal number:
  $$St = \frac{f \cdot D}{U}$$
  where $D$ is the effective diameter and $U$ is the flow velocity.
- **Multi-Modal Visualization:** Automatic generation of density ($|\psi|^2$), phase (vortex singularities), and vorticity fields.

## Installation

Ensure you have a JAX-compatible environment. For GPU/TPU support, follow the [official JAX installation guide](https://github.com/google/jax#installation).

```bash
pip install jax jaxlib matplotlib numpy
```

## Usage

The simulation is configured via the `Params` dataclass. You can run a parameter sweep (e.g., testing different velocities) directly from the script:

```python
# Set simulation grid and physics
params = Params(
    g=1.0, 
    dt=2e-4, 
    Nx=512, 
    Ny=384,
    obs_sigma=0.30
)

# Execute simulation
results = simulate_once(params, k0x=0.8, A=1.5, n_steps=30000)
```

## Results Example

The script outputs high-resolution heatmaps and line plots showing:
1. **Shedding Frequency vs Velocity:** Identifying the critical velocity for vortex nucleation.
2. **Vortex Population Dynamics:** Tracking the number of quantized vortices over time.
3. **Strouhal Maps:** Comparing superfluid hydrodynamics with classical fluid benchmarks.

## Author

**Hari Hardiyan**  
*AI & Physics Enthusiast*  
📧 [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)

I am passionate about the intersection of Artificial Intelligence and Computational Physics, specifically in leveraging machine learning frameworks like JAX to accelerate high-fidelity scientific simulations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

***
