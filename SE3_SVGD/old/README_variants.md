# Overview of SE3_SVGD Variants

This document provides a detailed explanation of the `SE3_SVGD` variants developed in this project, how they differ, and how they share underlying mechanics.

## 1. The High-Level Goal
All variants in this repository aim to solve the **Ergodic Trajectory Optimization** problem using **Stein Variational Gradient Descent (SVGD)**. The goal is to generate a set (ensemble) of trajectories for a robot or agent such that the time spent in different regions of the workspace is proportional to a target probability distribution (e.g., shapes like 'N', 'H', or 'II').

To measure this, all variants use the **Ergodic Metric**, which compares the spatial Fourier coefficients of the generated trajectories against the Fourier coefficients of the target distribution.

---

## 2. The "TSVEC" Variants (`tsvec_2d.py` & `tsvec_3d.py`)
TSVEC stands for *Time-Series Vector Ergodic Coverage*. These are the "regular" or "discrete" SVGD implementations.
*   **What is being optimized?** The SVGD algorithm directly optimizes the discrete spatial coordinates of the trajectory over time. A trajectory is represented simply as a matrix of shape $T \times \text{Dim}$ (e.g., $100 \times 2$ for 2D, or $100 \times 3$ for 3D).
*   **How it works:** In every iteration, the algorithm calculates the gradient of the energy function (Ergodic metric + smoothness + boundary constraints) with respect to every single waypoint in the trajectory. It then applies the SVGD update rule, using a repulsive kernel to ensure the trajectories in the ensemble push away from each other and explore different parts of the shape.
*   **Implementation:** These scripts primarily rely on **NumPy** for computing gradients analytically (e.g., `compute_smoothness_grad_numpy`, `fourier_basis_grad_nd`).

---

## 3. The "B-Spline SVGD" Variants (`svgd_bspline_2d.py` & `svgd_bspline_3d.py`)
These are the continuous, parameterized variants of the algorithm.
*   **What is being optimized?** Instead of optimizing the discrete trajectory waypoints, the SVGD algorithm optimizes a set of $K$ **B-Spline Control Points** (e.g., $K=29$ in 3D) and the initial starting state $s_0$.
*   **How it works:** 
    1. The B-Spline basis matrix translates the $K$ control points into a continuous trajectory.
    2. The trajectory is simulated forward in time to get discrete positions (using `forward_sim_nd`).
    3. The same energy function (Ergodic + smoothness + boundaries) is applied to these positions.
    4. The gradients are backpropagated *through the B-Spline forward simulation* all the way back to the control points.
*   **Implementation:** Because computing analytic gradients through a B-Spline simulation is incredibly complex, these scripts utilize **JAX** (`jax.grad`, `jax.jit`, `jax.vmap`). JAX automatically calculates the exact gradients of the control points relative to the energy function, allowing for a differentiable physics/trajectory engine.

---

## 4. The Shared Architecture (`ergodic_core.py` & `svgd_engine.py`)
To prevent massive code duplication between 2D, 3D, TSVEC, and B-Spline versions, the logic is highly centralized:

*   **`ergodic_core.py` (The Spatial Logic):** This is a purely dimension-agnostic file. It is responsible for taking your abstract target shapes ('N', 'H', 'II') and projecting them into spatial coordinates. It handles the new 3D projections (`plane`, `sphere`, `cube`), computes distances to these segments, and evaluates the Fourier basis functions.
*   **`svgd_engine.py` (The Optimization Logic):** This contains the actual loss functions (smoothness, boundary, obstacle, control regularization). Crucially, it houses both the JAX versions (`compute_smoothness_cost_jax`) used by the B-Spline scripts, and the NumPy versions (`compute_smoothness_cost_numpy`) used by the TSVEC scripts. It also houses the SVGD repulsive kernel logic.

---

## 5. 2D vs. 3D Differences
The leap from 2D to 3D mainly affects two things:
1.  **State Size:** 2D uses $[x, y]$ while 3D uses $[x, y, z]$. The B-Spline state vectors scale accordingly (e.g., from 4D kinematics to 6D kinematics).
2.  **Fourier Complexity:** The Fourier decomposition explodes in 3D. If you use $K=10$ wave numbers, 2D has $10^2 = 100$ modes. In 3D, $10^3 = 1000$ modes, which drastically slows down matrix multiplications. This is why `K_FOURIER` is often reduced (e.g., to 5, resulting in $125$ modes) in the 3D scripts to maintain computational feasibility.
