# Architectural Evolution & Results Analysis

This document traces the progression of the Conditional Flow Matching (CFM) architectures developed for learning Ergodic Coverage initializations. We analyze the design decisions, the hypotheses tested at each stage, and the resulting performance on **held-out (unseen)** trajectories.

The core objective is to generate B-Spline trajectories that adhere to a target density (condition) and to predict initial Lagrange multipliers ($\lambda_0$) for the downstream TSVEC solver.

---

## 1. Baseline: Additive Conditioning (`_char.py`)

### Architecture & Design Decisions
The initial approach utilized a standard 1D-CNN U-Net. The target shape (condition) was processed through a `ShapeEncoderMPD` which compressed the spatial information into a single feature vector using **Global Average Pooling (GAP)**. This vector was then simply *added* to the temporal features of the U-Net at various stages.

### What was tested?
This served as the baseline. The hypothesis was that a simple additive injection of a pooled feature vector would be sufficient for the U-Net to learn the mapping from a noisy trajectory to the target shape.

### Results & Analysis
> [!WARNING]
> **Mode Collapse and Information Loss**
> Simple additive conditioning provides a very weak gradient signal. More critically, the GAP layer acts as an extreme bottleneck. Averaging spatial features destroys the high-frequency structural details required for complex ergodic distributions. The model suffers from mode collapse, often predicting a "mean" shape or failing to capture the distinct topological features of the unseen held-out trajectories.

![Baseline Holdout Results](file:///c:/Users/Philipp/Documents/Uni/Master_Thesis/Master_thesis_ergodic_learning/thesis_architecture/Trajectory_data_generator/char_holdout_generation.png)

---

## 2. Stronger Injection: FiLM (`_char_film.py`)

### Architecture & Design Decisions
To combat the weak signal of additive conditioning, we introduced **Feature-wise Linear Modulation (FiLM)** using the `adaGN` (Adaptive GroupNorm) pattern. Instead of adding the pooled condition vector, the network uses it to predict a Scale ($\gamma$) and Shift ($\beta$) for the U-Net feature maps after every normalization layer. 

### What was tested?
The hypothesis was that modulating the variance and mean of the features dynamically based on the condition would force the network to pay more attention to the target shape. This is a standard practice in modern generative models (e.g., DiT).

### Results & Analysis
> [!NOTE]
> **Improved adherence, but bound by the GAP bottleneck**
> FiLM successfully forces the network to utilize the condition better. The generated trajectories adhere more closely to the general envelope of the target shapes. However, because the condition is *still* compressed via GAP before the FiLM projection, the network knows "how much" of a feature to use, but still lacks the fine-grained structural map. It struggles with sharp corners and complex intersections in the unseen holdouts.

![FiLM Holdout Results](file:///c:/Users/Philipp/Documents/Uni/Master_Thesis/Master_thesis_ergodic_learning/thesis_architecture/Trajectory_data_generator/char_holdout_generation_film.png)

---

## 3. Pushing the Limits: FiLM + CFG (`_char_film_cfg.py`)

### Architecture & Design Decisions
We retained the FiLM backbone but introduced **Classifier-Free Guidance (CFG)**. 
- **Training**: The condition is randomly dropped 10% of the time (`p_drop=0.1`) and replaced with a learnable null token.
- **Inference**: The model predicts both a conditioned flow ($v_{cond}$) and an unconditioned flow ($v_{uncond}$), extrapolating away from the unconditioned flow: $v_{cfg} = v_{uncond} + w \cdot (v_{cond} - v_{uncond})$.

### What was tested?
CFG amplifies the conditioning signal exponentially. The hypothesis was that by forcing the model to explicitly understand the difference between "generating a generic trajectory" and "generating *this specific* trajectory", we could squeeze the maximum possible performance out of the GAP-based architecture.

### Results & Analysis
> [!TIP]
> **Maximum performance for a pooled architecture**
> CFG yields a massive leap in sharpness and structural adherence. The model aggressively follows the conditioned shape, resulting in highly crisp trajectories. This represents the absolute limit of what a GAP-based architecture can achieve. However, it still occasionally hallucinates or smooths over extremely complex unseen topologies because the underlying frequency data was lost in the pooling layer.

![FiLM + CFG Holdout Results](file:///c:/Users/Philipp/Documents/Uni/Master_Thesis/Master_thesis_ergodic_learning/thesis_architecture/Trajectory_data_generator/char_holdout_generation_film_cfg.png)

---

## 4. The Paradigm Shift: Spectral Cross-Attention (`_spectral_crossattn.py`)

### Architecture & Design Decisions
A fundamental redesign to align with the downstream TSVEC solver, which relies on spectral metrics (Laplace-Beltrami/Fourier).
1. **Removed GAP**: We stopped pooling the condition.
2. **Spectral Tokenizer**: Converts $S$ spectral coefficients into $S$ distinct frequency tokens (with 2D positional encodings).
3. **Cross-Attention Bottleneck**: The U-Net bottleneck uses Multi-Head Cross-Attention. The B-Spline macro-tokens act as Queries, attending to the unpooled Spectral tokens (Keys/Values).
4. **Separation of Concerns**: FiLM is now used *exclusively* for Time conditioning, while Cross-Attention handles the Spatial/Frequency conditioning.
5. **Lambda Head**: Added a dual output to predict initial Lagrange multipliers ($\lambda_0$) directly from the bottleneck.

### What was tested?
The hypothesis was that ergodic coverage is fundamentally a frequency-matching problem. By feeding the network raw, unpooled spectral coefficients and allowing it to dynamically attend to specific frequency bands via cross-attention, it could perfectly map spectral constraints to spatial waypoints without any information loss.

### Results & Analysis
> [!IMPORTANT]
> **State-of-the-Art Generalization**
> The spectral architecture provides the best generalization to unseen holdouts. By maintaining the sequence of frequency tokens, the model can synthesize novel trajectories that respect high-frequency spatial components. The cross-attention mechanism allows the B-spline waypoints to "look up" exactly which frequency bands they need to satisfy at different stages of the trajectory. This eliminates mode collapse and sets the perfect foundation for predicting the TSVEC $\lambda_0$ constraints.

![Spectral Cross-Attention Holdout Results](file:///c:/Users/Philipp/Documents/Uni/Master_Thesis/Master_thesis_ergodic_learning/thesis_architecture/Trajectory_data_generator/char_holdout_generation_spectral.png)

---

## Conclusion
The progression demonstrates a clear trajectory from **weak injection** (Additive) to **strong injection** (FiLM + CFG), and finally to **structure-preserving injection** (Spectral Cross-Attention). 

The final spectral model not only produces the most robust held-out generations but also perfectly aligns with the mathematical foundations of the TSVEC solver it is designed to initialize.
