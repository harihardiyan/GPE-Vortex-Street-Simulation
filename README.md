
# Superfluid Vortex Dynamics: A Monolithic JAX Pipeline
### *High-Fidelity Quantum Hydrodynamics Simulation & Diagnostic Suite*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-9cf.svg?logo=google&logoColor=white)](https://jax.readthedocs.io/)
[![Field: Physics](https://img.shields.io/badge/Field-Quantum--Hydrodynamics-blue.svg)]()

## 📝 Overview

This project is a personal exploration into the fascinating world of **Bose-Einstein Condensates (BEC)** and **Superfluidity**. It implements a monolithic pipeline for solving the 2D Gross-Pitaevskii Equation (GPE) using high-performance, GPU-accelerated computing.

The "Monolith" is designed not just to simulate, but to **audit** and **analyze**. It tracks the emergence of quantum vortices behind obstacles, calculates drag forces with spectral precision, and verifies physical invariants (mass and energy) to ensure the simulation remains "physically honest." 

*Disclaimer: This tool was developed through a high-level orchestration of AI assistance and personal interest in computational physics. While I do not claim to be a physicist, the metrics and audits included are intended to meet rigorous research standards.*

---

## 🚀 Key Technical Features

### 1. The Numerical "Engine"
*   **Split-Step Fourier Method (SSFM):** Uses Strang splitting to ensure second-order accuracy in time, preserving the unitary nature of the wavefunction.
*   **ITP Relaxation:** Employs Imaginary Time Propagation to find the stable ground state (Thomas-Fermi profile) before starting real-time dynamics.
*   **JAX Acceleration:** Leverages XLA (Accelerated Linear Algebra) for JIT compilation, allowing the simulation to run at high speeds on CPUs, GPUs, or TPUs.

### 2. The "Audit" & Diagnostic Suite
*   **Energy & Norm Audits:** Continuous tracking of total energy ($E_{tot}$) and particle normalization ($N$) to detect numerical drift or instability.
*   **Mach-Strouhal Analysis:** Automatically calculates the **Mach Number ($M$)** based on measured upstream density and the **Strouhal Number ($St$)** via peak frequency detection in the drag power spectrum.
*   **Topological Defect Tracking:** An upstream-thresholded phase-winding counter identifies and counts quantized vortices by calculating the winding number around dense grid plaquettes.
*   **Absorbing Boundaries:** Uses "Sponge Layers" (Parabolic-Absorbing Potentials) to prevent wave reflection at the edges of the computational domain.

---

## 📐 The Physics Behind the Machine

The system solves the time-dependent **Gross-Pitaevskii Equation (GPE)** in a dimensionless form:

$$i \frac{\partial \psi(\mathbf{r},t)}{\partial t} = \left[ -\frac{1}{2}\nabla^2 + V_{ext}(\mathbf{r}) + g|\psi(\mathbf{r},t)|^2 \right] \psi(\mathbf{r},t)$$

### Critical Invariants
*   **The Sound Speed ($c_s$):** Calculated locally as $c_s = \sqrt{g \cdot n_{upstream}}$, defining the Mach number $M = U/c_s$.
*   **Quantum Vortex:** A topological defect where the phase $\phi$ winds by $2\pi$:
    $$\oint \nabla \phi \cdot d\ell = 2\pi n$$
*   **Drag Force ($F_x$):** Calculated by integrating the density against the gradient of the obstacle potential:
    $$F_x = -\int \rho(\mathbf{r}) \frac{\partial V_{obs}}{\partial x} d\mathbf{r}$$

---

## 🛠 Usage Guide

### Installation
Ensure you have a modern Python environment with JAX installed:
```bash
pip install jax jaxlib numpy matplotlib
```

### Running the Monolith
The script is designed to run a comprehensive sweep (e.g., searching for **Critical Velocity**):
```bash
python gpe_2d_q1_monolith_jax.py
```

### Customizing the Experiment
You can modify the `Params` dataclass within the script to change the physical environment:
*   `obs_height`: Strength of the obstacle.
*   `g`: Interaction strength (non-linearity).
*   `Lx, Ly`: Physical dimensions of the "Quantum Wind Tunnel."

---

## 📊 Outputs & Artifacts

The pipeline generates a `summary_monolith.json` and several high-fidelity plots:
1.  **Strouhal vs. Mach Plot:** Shows the transition from superfluidity to vortex shedding.
2.  **Drag Time-Series:** Tracks the oscillating force on the obstacle.
3.  **Phase Snapshots:** Uses the `twilight` cyclic colormap to visualize vortex cores and phase jumps.
4.  **Audit Logs:** Verification of Energy and Norm stability over time.

---

## 🤝 Citation & Acknowledgement

If you find this tool useful for academic reference or curiosity, please cite it as:

```text
Hardiyan, H. (2026). Superfluid Vortex Dynamics: A Monolithic JAX Diagnostic Pipeline. 
GitHub Repository: [Your-Repo-Link-Here]
```

## 📜 License

This project is licensed under the **MIT License**.

```text
Copyright (c) 2026 Hari Hardiyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---
*Disclaimer: This toolkit is provided as-is, born from a passion for the intersection of AI, programming, and the beauty of quantum fluids.*
