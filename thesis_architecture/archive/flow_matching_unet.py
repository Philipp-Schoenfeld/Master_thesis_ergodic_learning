import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import ErgodicTrajectoryDataset
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# UNet-style Conditional Flow Matching Network
#
# Architecture overview:
#
#   Input: x  (B, T, dim)   — noisy trajectory at flow time t
#          t  (B,)           — flow time scalar in [0, 1]
#          x0 (B, ctx_dim)   — conditioning context (start position)
#
#   Encoder (3 CNN layers with downsampling):
#     Conv1  : (B, dim + 1 + ctx_dim, T)  →  (B, C,   T)      [stride=1]
#     Conv2  : (B, C,   T)                →  (B, 2C, T/2)     [stride=2]
#     Conv3  : (B, 2C, T/2)              →  (B, 4C, T/4)     [stride=2]
#
#   Bottleneck MLP (applied per time-step in the lowest resolution):
#     Projects (4C) → (4C) with a small 2-layer MLP
#
#   Decoder (2 MLP layers with skip connections):
#     Up1 MLP : concat(4C, skip from Conv2-level 2C) = 6C  →  2C,  then upsample to T/2
#     Up2 MLP : concat(2C, skip from Conv1-level C)  = 3C  →  C,   then upsample to T
#
#   Output head:
#     Linear : C → dim   (predict velocity field v_t at each time-step)
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t ∈ [0,1] → R^{emb_dim} via sinusoidal encoding."""

    def __init__(self, emb_dim: int = 64):
        super().__init__()
        assert emb_dim % 2 == 0
        half = emb_dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half - 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
        t = t.view(-1)
        args = t[:, None] * self.freqs[None, :]        # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, emb_dim)
        return self.proj(emb)


class ConvBlock(nn.Module):
    """Conv1d + GroupNorm + SiLU, optionally with FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, cond_dim: int = 0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.SiLU()

        # FiLM: scale + shift conditioned on time/context embedding
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.film = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        # x: (B, C, T)
        h = self.norm(self.conv(x))
        if self.cond_dim > 0 and cond is not None:
            gamma_beta = self.film(cond)            # (B, out_ch*2)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = h * (1 + gamma[:, :, None]) + beta[:, :, None]
        return self.act(h)


class PointwiseMLP(nn.Module):
    """2-layer MLP applied independently to each point along the sequence axis."""

    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = None, cond_dim: int = 0):
        super().__init__()
        hidden_ch = hidden_ch or out_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden_ch),
            nn.SiLU(),
            nn.Linear(hidden_ch, out_ch),
        )
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.film = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        # x: (B, C, T) — apply MLP on C dimension for each t
        B, C, T = x.shape
        h = self.net(x.permute(0, 2, 1))        # (B, T, out_ch)
        h = h.permute(0, 2, 1)                   # (B, out_ch, T)
        if self.cond_dim > 0 and cond is not None:
            gamma_beta = self.film(cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = h * (1 + gamma[:, :, None]) + beta[:, :, None]
        return h


class UNetTrajectoryFlowNetwork(nn.Module):
    """
    UNet-style 1D conditional flow matching network.

    Encoder: 3 CNN layers (context embedding + downsampling)
    Decoder: 2 MLP layers with skip connections + upsampling

    Args:
        num_cps    : number of control points / trajectory length
        dim        : spatial dimension of each point (2 for 2D)
        context_dim: dimension of conditioning context (e.g. start position)
        base_ch    : base number of channels (encoder uses C, 2C, 4C)
        time_emb_dim: dimension of sinusoidal time embedding
    """

    def __init__(
        self,
        num_cps: int = 200,
        dim: int = 2,
        context_dim: int = 2,
        base_ch: int = 64,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.num_cps = num_cps
        self.dim = dim
        C = base_ch
        cond_dim = time_emb_dim + context_dim  # combined conditioning signal

        # --- Time embedding ---
        self.time_emb = SinusoidalTimeEmbedding(emb_dim=time_emb_dim)

        # --- Encoder: 3 CNN layers ---
        # Conv1: stride=1, full resolution T
        in_ch = dim  # trajectory channels (without conditioning; FiLM handles that)
        self.enc1 = ConvBlock(in_ch, C,    kernel_size=3, stride=1, cond_dim=cond_dim)
        # Conv2: stride=2, downsample T → T/2
        self.enc2 = ConvBlock(C,     C*2,  kernel_size=3, stride=2, cond_dim=cond_dim)
        # Conv3: stride=2, downsample T/2 → T/4
        self.enc3 = ConvBlock(C*2,   C*4,  kernel_size=3, stride=2, cond_dim=cond_dim)

        # --- Bottleneck MLP (per-point on lowest resolution) ---
        self.bottleneck = PointwiseMLP(C*4, C*4, hidden_ch=C*4, cond_dim=cond_dim)

        # --- Decoder: 2 MLP layers with skip connections ---
        # Up1: concat(bottleneck C*4, skip enc2 C*2) = C*6 → C*2
        self.up1 = PointwiseMLP(C*4 + C*2, C*2, hidden_ch=C*4, cond_dim=cond_dim)
        # Up2: concat(up1 C*2, skip enc1 C) = C*3 → C
        self.up2 = PointwiseMLP(C*2 + C,   C,   hidden_ch=C*2, cond_dim=cond_dim)

        # --- Output head ---
        self.out_head = nn.Conv1d(C, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, T, dim)  — noisy trajectory
            t       : (B,) or (B,1) — flow time
            context : (B, context_dim) — conditioning (start position)

        Returns:
            v : (B, T, dim) — predicted velocity field
        """
        B, T, D = x.shape

        # Conditioning signal: time emb + context
        t_emb = self.time_emb(t)               # (B, time_emb_dim)
        cond = torch.cat([t_emb, context], dim=-1)  # (B, cond_dim)

        # x to (B, dim, T) for Conv1d
        h = x.permute(0, 2, 1)                 # (B, D, T)

        # --- Encoder ---
        e1 = self.enc1(h,  cond)               # (B, C, T)
        e2 = self.enc2(e1, cond)               # (B, 2C, T/2)  ← skip connection
        e3 = self.enc3(e2, cond)               # (B, 4C, T/4)

        # --- Bottleneck ---
        b = self.bottleneck(e3, cond)          # (B, 4C, T/4)

        # --- Decoder Up1: upsample T/4 → T/2, concat with e2 ---
        b_up = F.interpolate(b,  size=e2.shape[-1], mode='linear', align_corners=False)
        d1 = torch.cat([b_up, e2], dim=1)     # (B, 6C, T/2)
        d1 = self.up1(d1, cond)               # (B, 2C, T/2)

        # --- Decoder Up2: upsample T/2 → T, concat with e1 ---
        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d1_up, e1], dim=1)    # (B, 3C, T)
        d2 = self.up2(d2, cond)               # (B, C, T)

        # --- Output ---
        v = self.out_head(d2)                  # (B, dim, T)
        return v.permute(0, 2, 1)             # (B, T, dim)


# ---------------------------------------------------------------------------
# Training utilities (same interface as flow_matching_trajectory_generation.py)
# ---------------------------------------------------------------------------

def compute_conditional_cfm_loss(model: nn.Module,
                                  x1_batch: torch.Tensor,
                                  context_batch: torch.Tensor) -> torch.Tensor:
    """Conditional Flow Matching loss (OT path: straight interpolation)."""
    batch_size, num_cps, dim = x1_batch.shape
    device = x1_batch.device

    x0 = torch.randn_like(x1_batch)
    t  = torch.rand(batch_size, device=device)

    t_exp = t.view(batch_size, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch   # interpolated
    ut    = x1_batch - x0                           # target vector field

    vt = model(xt, t, context_batch)
    return torch.mean((vt - ut) ** 2)


@torch.no_grad()
def generate_conditional_trajectories(
    model: nn.Module,
    context: torch.Tensor,
    num_samples: int,
    num_cps: int,
    dim: int,
    steps: int = 100,
    device: str = 'cpu',
):
    """Euler integration of the learned flow to generate trajectories."""
    model.eval()

    x = torch.randn(num_samples, num_cps, dim, device=device)
    dt = 1.0 / steps
    flow_history = [x.clone().cpu()]

    if context.dim() == 1:
        context = context.unsqueeze(0).repeat(num_samples, 1)
    context = context.to(device)

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        v = model(x, t, context)
        x = x + v * dt
        flow_history.append(x.clone().cpu())

    return x, flow_history


# ---------------------------------------------------------------------------
# Main: train and generate a test trajectory
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_cps      = 200
    dim          = 2
    context_dim  = 2
    base_ch      = 64
    time_emb_dim = 64

    model = UNetTrajectoryFlowNetwork(
        num_cps=num_cps,
        dim=dim,
        context_dim=context_dim,
        base_ch=base_ch,
        time_emb_dim=time_emb_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet model parameters: {total_params:,}")

    # --- Load ONLY the first trajectory (intentional overfitting) ---
    print("Loading single trajectory from database...")
    dataset = ErgodicTrajectoryDataset(
        '/home/philipp/Documents/Uni/Master_thesis/thesis_architecture/'
        'Trajectory_data_generator/stein_coverage_results.db',
        table_name='runs'
    )

    x1_single, ctx_single = dataset[0]          # (T, dim), (ctx_dim,)
    gt_traj   = x1_single.numpy()               # ground truth for later plotting
    gt_x0     = ctx_single.numpy()

    # Add batch dimension and move to device
    x1_single = x1_single.unsqueeze(0).to(device)    # (1, T, dim)
    ctx_single = ctx_single.unsqueeze(0).to(device)   # (1, ctx_dim)

    print(f"Single trajectory shape: {x1_single.shape}, context: {ctx_single.cpu().numpy()}")

    # --- Overfit training loop ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 200000
    log_every = 200

    print(f"Overfitting on a single trajectory for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_conditional_cfm_loss(model, x1_single, ctx_single)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

    # --- Generate trajectory using learned flow ---
    print("\nGenerating trajectory from learned flow...")
    context_tensor = ctx_single.squeeze(0)   # (ctx_dim,)

    gen_traj, _ = generate_conditional_trajectories(
        model,
        context=context_tensor,
        num_samples=1,
        num_cps=num_cps,
        dim=dim,
        steps=100,
        device=device,
    )
    gen_traj_np = gen_traj.cpu().numpy()[0]   # (T, dim)

    # -----------------------------------------------------------------------
    # Plot: Ground Truth vs Generated  |  Architecture diagram
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # --- Left: trajectory comparison ---
    ax = axes[0]
    ax.set_facecolor('white')
    ax.plot(gt_traj[:, 0], gt_traj[:, 1],
            color='#2196F3', linewidth=2.0, alpha=0.9, label='Target')
    ax.plot(gen_traj_np[:, 0], gen_traj_np[:, 1],
            color='#F44336', linewidth=2.0, alpha=0.9, label='Generated')
    ax.scatter([gt_x0[0]], [gt_x0[1]], color='black', s=80, zorder=5,
               label=f'x₀ = ({gt_x0[0]:.2f}, {gt_x0[1]:.2f})')
    ax.set_title("Overfit Sanity Check", fontsize=12, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.15, color='gray')
    ax.legend(frameon=False, fontsize=10)

    # --- Right: Architecture diagram ---
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("UNet Architecture", fontsize=12, pad=10)

    def box(ax, x, y, w, h, label, sublabel='', color='#EEEEEE', lw=1.2):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor='#555555', linewidth=lw,
                              zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + (0.10 if sublabel else 0), label, ha='center', va='center',
                fontsize=7.5, fontweight='bold', zorder=3, color='#222222')
        if sublabel:
            ax.text(x, y - 0.22, sublabel, ha='center', va='center',
                    fontsize=6.2, color='#666666', zorder=3)

    def arrow(ax, x0, y0, x1, y1, label=''):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.1),
                    zorder=1)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx + 0.1, my, label, fontsize=5.5, color='#888888',
                    ha='left', va='center')

    def skip_arrow(ax, x0, y0, x1, y1):
        """Curved skip connection arrow."""
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.9,
                                   connectionstyle='arc3,rad=-0.35'),
                    zorder=1)

    # Layout constants
    cx = 5.0           # center x for the main column
    col_r = 8.2        # x position of decoder column
    col_l = 1.8        # x position for skip labels

    bw, bh = 2.2, 0.7   # box width / height

    # ---- Input ----
    box(ax2, cx, 9.2, bw, bh, 'Input  x, t, x₀', '(B, T, dim) + scalar + (B, 2)', color='white')

    # ---- Time embedding ----
    box(ax2, cx, 8.1, bw, bh, 'Sin. Time Emb.', 'concat → cond (B, 128)', color='#F5F5F5')
    arrow(ax2, cx, 8.85, cx, 8.45)

    # ---- Encoder ----
    enc_ys  = [6.9, 5.8, 4.7]
    enc_lbl = ['Conv1d  ×1  stride=1', 'Conv1d  ×1  stride=2', 'Conv1d  ×1  stride=2']
    enc_sub = ['(B, C, T)  + GroupNorm + FiLM', '(B, 2C, T/2)  + GroupNorm + FiLM', '(B, 4C, T/4)  + GroupNorm + FiLM']
    enc_col = ['#E3F2FD', '#BBDEFB', '#90CAF9']

    arrow(ax2, cx, 7.75, cx, 7.25)
    for i, (ey, el, es, ec) in enumerate(zip(enc_ys, enc_lbl, enc_sub, enc_col)):
        box(ax2, cx, ey, bw, bh, el, es, color=ec)
        if i < len(enc_ys) - 1:
            arrow(ax2, cx, ey - bh/2, cx, enc_ys[i+1] + bh/2)

    # ---- Bottleneck ----
    arrow(ax2, cx, 4.35, cx, 3.85)
    box(ax2, cx, 3.5, bw, bh, 'Bottleneck MLP  ×2', '(B, 4C, T/4)  + FiLM', color='#E8EAF6')

    # ---- Decoder (right column) ----
    dec_ys  = [2.4, 1.3]
    dec_lbl = ['Upsample + cat(skip enc2)', 'Upsample + cat(skip enc1)']
    dec_sub = ['MLP ×2  →  (B, 2C, T/2)  + FiLM', 'MLP ×2  →  (B, C, T)      + FiLM']
    dec_col = ['#E8F5E9', '#C8E6C9']

    arrow(ax2, cx, 3.15, cx, 2.75)
    for i, (dy, dl, ds, dc) in enumerate(zip(dec_ys, dec_lbl, dec_sub, dec_col)):
        box(ax2, cx, dy, bw + 0.4, bh, dl, ds, color=dc)
        if i < len(dec_ys) - 1:
            arrow(ax2, cx, dy - bh/2, cx, dec_ys[i+1] + bh/2)

    # ---- Output head ----
    arrow(ax2, cx, 0.95, cx, 0.55)
    box(ax2, cx, 0.25, bw, bh, 'Output  Conv1d 1×1', 'v  (B, T, dim)  velocity field', color='white')

    # ---- Skip connections (arcs from enc → dec) ----
    # enc2 (y=5.8) → up1 (y=2.4): right side
    skip_arrow(ax2, cx + bw/2, 5.8, cx + (bw+0.4)/2, 2.4)
    ax2.text(8.5, 4.1, 'skip e2', fontsize=5.5, color='#999999', ha='center', va='center')
    # enc1 (y=6.9) → up2 (y=1.3): right side (wider arc)
    ax2.annotate('', xy=(cx + (bw+0.4)/2, 1.3), xytext=(cx + bw/2, 6.9),
                 arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=0.9,
                                 connectionstyle='arc3,rad=-0.28'), zorder=1)
    ax2.text(9.2, 4.1, 'skip e1', fontsize=5.5, color='#BBBBBB', ha='center', va='center')

    plt.suptitle("UNet Flow Matching — Overfitting Sanity Check", fontsize=13,
                 color='#333333', y=1.01)
    plt.tight_layout()
    plt.show()

