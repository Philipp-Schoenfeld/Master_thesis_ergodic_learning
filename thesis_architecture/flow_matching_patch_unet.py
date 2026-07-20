import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src/ to path so the init_strategies package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from init_strategies.polynomial import init_particles as polynomial_init_particles


# ---------------------------------------------------------------------------
# Patchify U-Net Flow Matching Network
#
# Architecture overview:
#
#   Input: x  (B, nξ, nd)  — noisy B-spline control points at flow time t
#          t  (B,)          — ODE integration time scalar in [0, 1]
#
#   Stage 1 — Patchification (Tokenisation):
#     1D-CNN  kernel=nd, stride=nd  →  tokens (B, D, nξ)
#     Translates raw nd-dim coordinates into D-dim embeddings.
#
#   Stage 2 — Time Conditioning:
#     Sinusoidal embedding  t ∈ R  →  t′ ∈ R^D
#     Added (broadcast) to every token: (B, D, nξ)
#
#   Stage 3 — 1D U-Net Backbone (internal communication only):
#     Encoder:
#       Enc1: Conv1d + GroupNorm + SiLU   (B, D,   nξ)   [stride=1]
#       Enc2: Conv1d + GroupNorm + SiLU   (B, 2D, nξ/2)  [stride=2]
#       Enc3: Conv1d + GroupNorm + SiLU   (B, 4D, nξ/4)  [stride=2]
#     Bottleneck:
#       Conv1d × 2                         (B, 4D, nξ/4)
#     Decoder (skip connections from encoder):
#       Dec1: Upsample + cat(skip enc2) → Conv1d  (B, 2D, nξ/2)
#       Dec2: Upsample + cat(skip enc1) → Conv1d  (B, D,  nξ)
#
#   Stage 4 — Output MLP Head (de-tokenisation):
#     Linear + LayerNorm + Linear  (B, nξ, D) → (B, nξ, nd)
#     Reconstructs nd-dim velocity vector per control point.
#
#   Output: vθ (B, nξ, nd) — predicted velocity field
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Patchification Layer
# ---------------------------------------------------------------------------

class PatchifyLayer(nn.Module):
    """
    Tokenisation via 1D CNN.

    Maps raw nd-dimensional control-point coordinates → D-dimensional tokens.

    The description specifies a kernel of (1, nd): width-1 over the sequence
    axis, spanning all nd feature channels of a single control point.  In
    Conv1d terms (where nd is the channel dimension and nξ is the sequence
    length) this is:

        Conv1d(in_channels=nd, out_channels=D, kernel_size=1, stride=1)

    — i.e. a pointwise linear projection applied independently at each of the
    nξ positions, inflating nd → D without mixing neighbouring control points.
    This exactly preserves the output length nξ.
    """

    def __init__(self, nd: int, D: int):
        super().__init__()
        self.nd = nd
        self.D  = D
        # kernel_size=1, stride=1: pointwise projection nd → D per control point
        self.conv = nn.Conv1d(
            in_channels=nd,
            out_channels=D,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, nξ, nd)  — B-spline control points
        Returns:
            tokens: (B, D, nξ)  — D-dim embedding per control point
        """
        # Permute to (B, nd, nξ) for Conv1d, then project each position nd → D
        h = self.conv(x.permute(0, 2, 1))               # (B, D, nξ)
        # LayerNorm over the channel (D) dimension
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)  # (B, D, nξ)
        return h


# ---------------------------------------------------------------------------
# 2. Sinusoidal Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t ∈ [0, 1]  →  R^D via sinusoidal positional encoding.

    Frequencies follow the standard transformer schedule:
        freq_i = 1 / 10000^(2i / D)

    The resulting embedding is projected back to R^D via a small MLP so the
    network can learn an affine transformation of the raw encoding.
    """

    def __init__(self, D: int):
        super().__init__()
        assert D % 2 == 0, "D must be even for sinusoidal embedding."
        half = D // 2
        # Pre-compute fixed log-spaced frequencies
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * -(torch.log(torch.tensor(10_000.0)) / (half - 1))
        )
        self.register_buffer("freqs", freqs)   # (half,)
        self.proj = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.SiLU(),
            nn.Linear(D * 2, D),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1)
        Returns:
            t_emb: (B, D)
        """
        t = t.view(-1)                                      # (B,)
        args = t[:, None] * self.freqs[None, :]             # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, D)
        return self.proj(emb)                                # (B, D)


# ---------------------------------------------------------------------------
# 3. 1D U-Net building blocks
# ---------------------------------------------------------------------------

class ConvResBlock(nn.Module):
    """
    Residual 1D-CNN block: two Conv1d + GroupNorm + SiLU layers with a
    residual skip. The time embedding is added (broadcast) before the second
    activation — no FiLM is needed because there is no external conditioning
    signal, only the global time t.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,
                               padding=pad)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act   = nn.SiLU()

        # 1×1 projection for the residual path when channels or stride differ
        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, L)"""
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.residual(x))


class UNetBackbone(nn.Module):
    """
    Pure 1D U-Net backbone operating on token sequences (B, D, nξ).

    The 1D kernels provide a local inductive bias that intrinsically
    encourages kinematic smoothness (C^k continuity) along the sequence of
    B-spline control points — no external conditioning, no cross-attention.

    Encoder:
        Enc1: D   → D   [stride=1]
        Enc2: D   → 2D  [stride=2]
        Enc3: 2D  → 4D  [stride=2]
    Bottleneck:
        4D → 4D (two residual blocks)
    Decoder:
        Dec1: upsample + cat(skip enc2)  6D → 2D
        Dec2: upsample + cat(skip enc1)  3D → D
    """

    def __init__(self, D: int, kernel_size: int = 3):
        super().__init__()
        # --- Encoder ---
        self.enc1 = ConvResBlock(D,   D,   kernel_size, stride=1)
        self.enc2 = ConvResBlock(D,   D*2, kernel_size, stride=2)
        self.enc3 = ConvResBlock(D*2, D*4, kernel_size, stride=2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ConvResBlock(D*4, D*4, kernel_size, stride=1),
            ConvResBlock(D*4, D*4, kernel_size, stride=1),
        )

        # --- Decoder ---
        # After upsample + concat: (D*4 + D*2) = D*6  →  D*2
        self.dec1 = ConvResBlock(D*4 + D*2, D*2, kernel_size, stride=1)
        # After upsample + concat: (D*2 + D)  = D*3  →  D
        self.dec2 = ConvResBlock(D*2 + D,   D,   kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, nξ)  — time-conditioned token sequence
        Returns:
            out: (B, D, nξ)
        """
        # Encoder
        e1 = self.enc1(x)   # (B, D,   nξ)
        e2 = self.enc2(e1)  # (B, 2D, nξ/2)
        e3 = self.enc3(e2)  # (B, 4D, nξ/4)

        # Bottleneck
        b = self.bottleneck(e3)  # (B, 4D, nξ/4)

        # Decoder — upsample to match skip connection length, then concat
        b_up = F.interpolate(b,  size=e2.shape[-1], mode='linear',
                             align_corners=False)              # (B, 4D, nξ/2)
        d1 = self.dec1(torch.cat([b_up, e2], dim=1))          # (B, 2D, nξ/2)

        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear',
                              align_corners=False)             # (B, 2D, nξ)
        d2 = self.dec2(torch.cat([d1_up, e1], dim=1))         # (B, D,  nξ)

        return d2


# ---------------------------------------------------------------------------
# 4. Output MLP Head (de-tokenisation / de-patchification)
# ---------------------------------------------------------------------------

class OutputMLPHead(nn.Module):
    """
    Projects D-dimensional tokens back to nd-dimensional velocity vectors.

    Applied point-wise (independently to each token along the nξ axis):
        (B, nξ, D) → Linear → LayerNorm → SiLU → Linear → (B, nξ, nd)

    This is the inverse of the patchification step in concept — it
    "de-embeds" the latent representation into a precise nd velocity vector
    per control point.
    """

    def __init__(self, D: int, nd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.SiLU(),
            nn.Linear(D, nd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, nξ)
        Returns:
            v: (B, nξ, nd)
        """
        # Permute to (B, nξ, D) for pointwise Linear, then back
        return self.net(x.permute(0, 2, 1))   # (B, nξ, nd)


# ---------------------------------------------------------------------------
# 5. Full Model
# ---------------------------------------------------------------------------

class PatchUNetFlowNetwork(nn.Module):
    """
    Patchify U-Net flow matching network for B-spline trajectory generation.

    Pipeline:
        x (B, nξ, nd)  →  PatchifyLayer  →  tokens (B, D, nξ)
                       →  + SinusoidalTimeEmbedding(t)  [broadcast per token]
                       →  UNetBackbone  →  (B, D, nξ)
                       →  OutputMLPHead  →  vθ (B, nξ, nd)

    No cross-attention, no external environment features — suitable for
    obstacle-free ergodic trajectory generation where t is the sole
    conditioning signal.

    Args:
        nxi : number of B-spline control points  (nξ)
        nd  : spatial dimension per control point (e.g. 3 for 3-D)
        D   : embedding / token dimension         (e.g. 256)
        kernel_size: 1D-CNN kernel size in the U-Net backbone (default 3)
    """

    def __init__(
        self,
        nxi: int = 200,
        nd: int  = 3,
        D: int   = 256,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D

        # Stage 1: Patchification
        self.patchify = PatchifyLayer(nd=nd, D=D)

        # Stage 2: Sinusoidal time embedding
        self.time_emb = SinusoidalTimeEmbedding(D=D)

        # Stage 3: 1D U-Net backbone
        self.backbone = UNetBackbone(D=D, kernel_size=kernel_size)

        # Stage 4: Output MLP head
        self.head = OutputMLPHead(D=D, nd=nd)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, nξ, nd)  — noisy B-spline control points
            t : (B,) or (B, 1) — ODE integration time in [0, 1]

        Returns:
            vθ : (B, nξ, nd)  — predicted velocity field (one nd-vector per CP)
        """
        # Stage 1 — Patchification: raw coords → token embeddings
        tokens = self.patchify(x)                  # (B, D, nξ)

        # Stage 2 — Time conditioning: broadcast t′ to every token position
        t_emb = self.time_emb(t)                   # (B, D)
        tokens = tokens + t_emb[:, :, None]         # (B, D, nξ)  — additive injection

        # Stage 3 — U-Net backbone: internal spatial communication
        tokens = self.backbone(tokens)             # (B, D, nξ)

        # Stage 4 — Output head: de-tokenise back to velocity space
        vtheta = self.head(tokens)                 # (B, nξ, nd)
        return vtheta


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_cfm_loss(model: nn.Module,
                     x1_batch: torch.Tensor) -> torch.Tensor:
    """
    Conditional Flow Matching loss (OT straight-line path).

    No conditioning context — the only signal is the flow time t.

    Args:
        model   : PatchUNetFlowNetwork
        x1_batch: (B, nξ, nd)  — target (clean) B-spline control points
    Returns:
        scalar MSE loss
    """
    B, nxi, nd = x1_batch.shape
    device = x1_batch.device

    x0 = torch.randn_like(x1_batch)                          # source noise
    t  = torch.rand(B, device=device)                         # random flow time

    t_exp = t.view(B, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch              # interpolated path
    ut    = x1_batch - x0                                     # target velocity

    vt = model(xt, t)
    return torch.mean((vt - ut) ** 2)


@torch.no_grad()
def generate_trajectories(
    model: nn.Module,
    num_samples: int,
    nxi: int,
    nd: int,
    steps: int = 100,
    device: str = 'cpu',
):
    """
    Euler integration of the learned flow from noise x0 ~ N(0,I) to data.

    Args:
        model      : trained PatchUNetFlowNetwork
        num_samples: number of trajectories to generate
        nxi        : number of control points
        nd         : spatial dimension
        steps      : number of Euler steps
        device     : torch device string

    Returns:
        x           : (num_samples, nξ, nd)  — final generated trajectories
        flow_history: list of (num_samples, nξ, nd) tensors at each step
    """
    model.eval()
    x = torch.randn(num_samples, nxi, nd, device=device)
    dt = 1.0 / steps
    flow_history = [x.clone().cpu()]

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        v = model(x, t)
        x = x + v * dt
        flow_history.append(x.clone().cpu())

    return x, flow_history


# ---------------------------------------------------------------------------
# Main: overfit on single trajectory + visualise
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyper-parameters
    nxi         = 200   # number of B-spline control points
    nd          = 2     # spatial dimension (2-D for the dataset)
    D           = 256   # token / embedding dimension
    kernel_size = 3     # 1D U-Net kernel size

    model = PatchUNetFlowNetwork(
        nxi=nxi,
        nd=nd,
        D=D,
        kernel_size=kernel_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchUNet model parameters: {total_params:,}")

    # --- Generate training data via polynomial N-shape initialisation ---
    N_particles = 32          # batch size: number of noisy N-trajectories
    noise_std   = 0.02        # additive Gaussian noise on the base trajectory
    poly_degree = 5           # degree of the parametric polynomial fit

    print(f"Generating {N_particles} polynomial N-shape trajectories "
          f"(T={nxi}, noise_std={noise_std}, degree={poly_degree})...")

    particles_np, base_traj = polynomial_init_particles(
        N=N_particles, T=nxi,
        degree=poly_degree, noise_std=noise_std
    )
    # particles_np: (N, nxi*nd) — ravelled; reshape to (N, nxi, nd)
    x1_batch = torch.tensor(
        particles_np.reshape(N_particles, nxi, nd),
        dtype=torch.float32
    ).to(device)              # (N, nξ, nd)

    print(f"Training batch shape: {x1_batch.shape}")

    # --- Training loop ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs    = 5_000
    log_every = 500

    print(f"Training on polynomial N-init batch for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_cfm_loss(model, x1_batch)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:6d} | Loss: {loss.item():.6f}")

    # --- Generate trajectories using learned flow ---
    n_gen = 1     # how many samples to generate for visualisation
    print(f"\nGenerating {n_gen} trajectories from learned flow...")
    gen_trajs, _ = generate_trajectories(
        model,
        num_samples=n_gen,
        nxi=nxi,
        nd=nd,
        steps=100,
        device=device,
    )
    gen_trajs_np = gen_trajs.cpu().numpy()   # (n_gen, nξ, nd)

    # -----------------------------------------------------------------------
    # Plot: Training data + Generated samples  |  Architecture diagram
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')

    # --- Left: training data vs generated samples ---
    ax = axes[0]
    ax.set_facecolor('white')
    # Plot a few training samples (grey)
    for i in range(min(8, N_particles)):
        traj = particles_np[i].reshape(nxi, nd)
        ax.plot(traj[:, 0], traj[:, 1],
                color='#90CAF9', linewidth=1.0, alpha=0.5,
                label='Training sample' if i == 0 else '')
    # Base (noiseless) N-shape
    ax.plot(base_traj[:, 0], base_traj[:, 1],
            color='#1565C0', linewidth=2.5, alpha=0.95, label='Base N-traj')
    # Generated samples
    for i, gt in enumerate(gen_trajs_np):
        ax.plot(gt[:, 0], gt[:, 1],
                color='#EF5350', linewidth=1.2, alpha=0.7,
                label='Generated' if i == 0 else '')
    ax.set_title("Polynomial N-Init  →  Flow Matching", fontsize=12, pad=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.15, color='gray')
    ax.legend(frameon=False, fontsize=9)

    # --- Right: Architecture diagram ---
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 11)
    ax2.axis('off')
    ax2.set_title("PatchUNet Architecture", fontsize=12, pad=10)

    def box(ax, x, y, w, h, label, sublabel='', color='#EEEEEE', lw=1.2):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor='#555555',
                              linewidth=lw, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + (0.12 if sublabel else 0), label, ha='center',
                va='center', fontsize=7.5, fontweight='bold', zorder=3,
                color='#222222')
        if sublabel:
            ax.text(x, y - 0.22, sublabel, ha='center', va='center',
                    fontsize=6.0, color='#555555', zorder=3)

    def arrow(ax, x0, y0, x1, y1, label=''):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.1),
                    zorder=1)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx+0.12, my, label, fontsize=5.5, color='#888888',
                    ha='left', va='center')

    def skip_arrow(ax, x0, y0, x1, y1, rad=-0.35, color='#999999'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.9,
                                   connectionstyle=f'arc3,rad={rad}'),
                    zorder=1)

    cx  = 5.0          # center x of main column
    bw  = 2.6          # box width
    bh  = 0.70         # box height

    # Input
    box(ax2, cx, 10.3, bw, bh, 'Input  x',
        '(B, nξ, nd)  — noisy control points', color='white')
    arrow(ax2, cx, 9.95, cx, 9.45)

    # Stage 1 — Patchification
    box(ax2, cx, 9.10, bw, bh, '① Patchify  (1D-CNN)',
        'kernel=stride=nd  →  (B, D, nξ)', color='#FFF8E1')
    arrow(ax2, cx, 8.75, cx, 8.25)

    # Stage 2 — Time embedding
    box(ax2, cx, 7.90, bw, bh, '② Sin. Time Embedding',
        't ∈ R  →  t′ ∈ R^D  (broadcast +)', color='#F3E5F5')
    arrow(ax2, cx, 7.55, cx, 7.05)

    # Stage 3 — Encoder
    enc_ys   = [6.70, 5.65, 4.60]
    enc_lbls = ['Enc1  Conv1d  stride=1',
                'Enc2  Conv1d  stride=2',
                'Enc3  Conv1d  stride=2']
    enc_subs = ['(B, D,   nξ)  ResBlock + GroupNorm',
                '(B, 2D, nξ/2) ResBlock + GroupNorm',
                '(B, 4D, nξ/4) ResBlock + GroupNorm']
    enc_cols = ['#E3F2FD', '#BBDEFB', '#90CAF9']

    for i, (ey, el, es, ec) in enumerate(zip(enc_ys, enc_lbls, enc_subs, enc_cols)):
        box(ax2, cx, ey, bw, bh, el, es, color=ec)
        if i < len(enc_ys) - 1:
            arrow(ax2, cx, ey - bh/2, cx, enc_ys[i+1] + bh/2)

    # Bottleneck
    arrow(ax2, cx, 4.25, cx, 3.75)
    box(ax2, cx, 3.40, bw, bh, 'Bottleneck  Conv1d × 2',
        '(B, 4D, nξ/4)  ResBlock × 2', color='#E8EAF6')

    # Stage 3 — Decoder
    dec_ys   = [2.35, 1.30]
    dec_lbls = ['Dec1  Upsample + cat(skip Enc2)',
                'Dec2  Upsample + cat(skip Enc1)']
    dec_subs = ['Conv1d  →  (B, 2D, nξ/2)  ResBlock',
                'Conv1d  →  (B, D,  nξ)    ResBlock']
    dec_cols = ['#E8F5E9', '#C8E6C9']

    arrow(ax2, cx, 3.05, cx, 2.70)
    for i, (dy, dl, ds, dc) in enumerate(zip(dec_ys, dec_lbls, dec_subs, dec_cols)):
        box(ax2, cx, dy, bw + 0.3, bh, dl, ds, color=dc)
        if i < len(dec_ys) - 1:
            arrow(ax2, cx, dy - bh/2, cx, dec_ys[i+1] + bh/2)

    # Stage 4 — Output MLP head
    arrow(ax2, cx, 0.95, cx, 0.55)
    box(ax2, cx, 0.20, bw, bh, '④ Output MLP Head',
        'Linear → LN → SiLU → Linear  →  vθ (B, nξ, nd)',
        color='#FFF3E0')

    # Skip connections (arcs on the right side)
    # Enc2 → Dec1
    skip_arrow(ax2, cx + bw/2, 5.65, cx + (bw+0.3)/2, 2.35, rad=-0.3)
    ax2.text(8.6, 4.0, 'skip e2', fontsize=5.5, color='#888888',
             ha='center', va='center')
    # Enc1 → Dec2
    ax2.annotate('', xy=(cx + (bw+0.3)/2, 1.30),
                 xytext=(cx + bw/2, 6.70),
                 arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=0.9,
                                 connectionstyle='arc3,rad=-0.25'), zorder=1)
    ax2.text(9.2, 4.0, 'skip e1', fontsize=5.5, color='#BBBBBB',
             ha='center', va='center')

    plt.suptitle(
        f"PatchUNet Flow Matching — Overfitting Sanity Check\n"
        f"nξ={nxi}, nd={nd}, D={D}, params={total_params:,}",
        fontsize=12, color='#333333', y=1.01
    )
    plt.tight_layout()
    plt.show()
