import torch
import torch.nn as nn
from data_loader import ErgodicTrajectoryDataset
from torch.utils.data import DataLoader


class ConditionalTrajectoryFlowNetwork(nn.Module):
    def __init__(self, num_control_points, dim, context_dim=2, hidden_dim=256):
        super().__init__()
        self.num_cps = num_control_points
        self.dim = dim

        input_size = (num_control_points * dim) + 1 + context_dim
        output_size = num_control_points * dim

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x, t, context):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        xtc = torch.cat([x_flat, t, context], dim=-1)
        v_flat = self.net(xtc)
        return v_flat.view(batch_size, self.num_cps, self.dim)


class PointwiseMLPFlowNetwork(nn.Module):
    """MLP applied independently at each control point. Input per point: [t | x_i | context]."""
    def __init__(self, dim=2, context_dim=2, hidden_dim=256, num_layers=5):
        super().__init__()
        self.dim = dim

        input_dim = dim + 1 + context_dim
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim))  # no activation on output
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, context):
        batch_size, num_cps, _ = x.shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_exp   = t.unsqueeze(1).expand(-1, num_cps, -1)
        ctx_exp = context.unsqueeze(1).expand(-1, num_cps, -1)
        inp = torch.cat([t_exp, x, ctx_exp], dim=-1)
        v_flat = self.net(inp.view(batch_size * num_cps, -1))
        return v_flat.view(batch_size, num_cps, self.dim)


class FiLMConvBlock(nn.Module):
    """Conv1d residual block with FiLM conditioning. Uses GroupNorm for stability at small batch sizes."""
    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.act  = nn.SiLU()
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.conv(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return x + h


class AxialAttentionBlock(nn.Module):
    """Self-attention over the sequence dimension, FiLM-conditioned. Lets the model relate distant control points."""
    def __init__(self, channels: int, cond_dim: int, n_heads: int = 4, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return x + h


class FiLMCNNAttentionFlowNetwork(nn.Module):
    """
    Velocity-field network for Conditional Flow Matching.
    Layout: stem -> [conv, conv, attn] x2 -> conv, conv -> head
    Conditioning cond = [t | context] is injected via FiLM into every block.
    """
    def __init__(self, dim: int = 2, context_dim: int = 2, hidden_channels: int = 64, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        C = hidden_channels
        cond_dim = 1 + context_dim
        in_ch = dim + 1 + context_dim

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, C, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.g1_conv1 = FiLMConvBlock(C, cond_dim)
        self.g1_conv2 = FiLMConvBlock(C, cond_dim)
        self.g1_attn  = AxialAttentionBlock(C, cond_dim, n_heads)

        self.g2_conv1 = FiLMConvBlock(C, cond_dim)
        self.g2_conv2 = FiLMConvBlock(C, cond_dim)
        self.g2_attn  = AxialAttentionBlock(C, cond_dim, n_heads)

        self.tail_conv1 = FiLMConvBlock(C, cond_dim)
        self.tail_conv2 = FiLMConvBlock(C, cond_dim)

        self.head = nn.Conv1d(C, dim, kernel_size=1)  # no activation

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, num_cps, _ = x.shape

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        cond = torch.cat([t, context], dim=-1)
        cond_seq = cond.unsqueeze(-1).expand(-1, -1, num_cps)
        inp = torch.cat([x.permute(0, 2, 1), cond_seq], dim=1)

        h = self.stem(inp)

        h = self.g1_conv1(h, cond)
        h = self.g1_conv2(h, cond)
        h = self.g1_attn(h, cond)

        h = self.g2_conv1(h, cond)
        h = self.g2_conv2(h, cond)
        h = self.g2_attn(h, cond)

        h = self.tail_conv1(h, cond)
        h = self.tail_conv2(h, cond)

        return self.head(h).permute(0, 2, 1)


class CNNMLPTrajectoryFlowNetwork(nn.Module):
    """3 CNN layers for local feature extraction, 2 MLP layers for global integration."""
    def __init__(self, num_control_points=200, dim=2, context_dim=2, hidden_channels=64, hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.num_cps = num_control_points

        in_channels = dim + 1 + context_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        cnn_out_dim = hidden_channels * num_control_points
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_control_points * dim),
        )

    def forward(self, x, t, context):
        batch_size, num_cps, _ = x.shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        cond = torch.cat([t, context], dim=-1)
        cond_expanded = cond.unsqueeze(-1).expand(-1, -1, num_cps)
        cnn_input = torch.cat([x.permute(0, 2, 1), cond_expanded], dim=1)
        cnn_out = self.cnn(cnn_input)
        v_flat = self.mlp(cnn_out.view(batch_size, -1))
        return v_flat.view(batch_size, self.num_cps, self.dim)


class CNNTrajectoryFlowNetwork(nn.Module):
    def __init__(self, dim=2, context_dim=2, hidden_channels=64):
        super().__init__()
        self.dim = dim
        in_channels = dim + 1 + context_dim
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, dim, kernel_size=3, padding=1)
        )

    def forward(self, x, t, context):
        batch_size, num_cps, _ = x.shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        cond = torch.cat([t, context], dim=-1)
        cond_expanded = cond.unsqueeze(-1).expand(-1, -1, num_cps)
        cnn_input = torch.cat([x.permute(0, 2, 1), cond_expanded], dim=1)
        return self.conv_net(cnn_input).permute(0, 2, 1)


def compute_conditional_cfm_loss(model, x1_batch, context_batch):
    batch_size, num_cps, dim = x1_batch.shape
    device = x1_batch.device

    x0 = torch.randn_like(x1_batch)
    t = torch.rand(batch_size, 1, device=device)

    t_expanded = t.view(batch_size, 1, 1)
    xt = (1 - t_expanded) * x0 + t_expanded * x1_batch
    ut = x1_batch - x0

    vt = model(xt, t, context_batch)
    return torch.mean((vt - ut) ** 2)


@torch.no_grad()
def generate_conditional_trajectories(model, context, num_samples, num_cps, dim, steps=100, device='cpu'):
    model.eval()

    x = torch.randn(num_samples, num_cps, dim, device=device)
    dt = 1.0 / steps
    flow_history = [x.clone().cpu()]

    if context.dim() == 1:
        context = context.unsqueeze(0).repeat(num_samples, 1)
    context = context.to(device)

    for step in range(steps):
        t = torch.full((num_samples, 1), step * dt, device=device)
        v = model(x, t, context)
        x = x + v * dt
        flow_history.append(x.clone().cpu())

    return x, flow_history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_cps = 200
    dim = 2
    context_dim = 2

    model = FiLMCNNAttentionFlowNetwork(
        dim=dim,
        context_dim=context_dim,
        hidden_channels=64,
        n_heads=4,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: FiLMCNNAttentionFlowNetwork  |  Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Loading dataset from database...")
    dataset = ErgodicTrajectoryDataset(
        '/home/philipp/Documents/Uni/Master_thesis/thesis_architecture/Trajectory_data_generator/stein_coverage_results.db',
        table_name='runs'
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"Starting conditional training on {len(dataset)} trajectories...")
    model.train()
    epochs = 1000

    for epoch in range(epochs):
        total_loss = 0.0
        for x1_batch, context_batch in dataloader:
            x1_batch     = x1_batch.to(device)
            context_batch = context_batch.to(device)

            optimizer.zero_grad()
            loss = compute_conditional_cfm_loss(model, x1_batch, context_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    print("Generating conditional test trajectories...")
    test_context = torch.tensor([0.1, 0.2], dtype=torch.float32)

    final_trajs, history = generate_conditional_trajectories(
        model,
        context=test_context,
        num_samples=1,
        num_cps=num_cps,
        dim=dim,
        steps=50,
        device=device
    )

    final_trajs_np = final_trajs.cpu().numpy()

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax_traj, ax_arch) = plt.subplots(1, 2, figsize=(16, 8))

    ax_traj.plot(
        final_trajs_np[0, :, 0], final_trajs_np[0, :, 1],
        marker='o', markersize=2, linestyle='-', linewidth=1.2,
        color='blue', label='Generated trajectory'
    )
    ax_traj.scatter(
        [test_context[0].item()], [test_context[1].item()],
        color='red', s=100, zorder=5, label=f'Conditioned Start (x0) = {test_context.tolist()}'
    )
    ax_traj.set_title(f'Conditioned Trajectory Generation (Start: {test_context.tolist()})')
    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')
    ax_traj.legend()
    ax_traj.grid(True)

    ax_arch.set_xlim(0, 1)
    ax_arch.set_ylim(-0.05, 1.02)
    ax_arch.axis('off')
    ax_arch.set_title('FiLMCNNAttentionFlowNetwork – Architecture')

    C = 64
    arch_layers = [
        (f'Input  [x({dim}) | t(1) | ctx({context_dim})]  →  {dim+1+context_dim} ch × {num_cps} pts', 0.93),
        (f'Stem:  Conv1d({dim+1+context_dim}→{C})  +  SiLU',                                           0.84),
        (f'FiLMConvBlock  Conv1d({C}→{C})  +  GN  +  SiLU  +  FiLM  +  residual',                     0.75),
        (f'FiLMConvBlock  Conv1d({C}→{C})  +  GN  +  SiLU  +  FiLM  +  residual',                     0.66),
        (f'AxialAttentionBlock  MHA({C}, heads=4)  +  FiLM  +  residual',                              0.57),
        (f'FiLMConvBlock  Conv1d({C}→{C})  +  GN  +  SiLU  +  FiLM  +  residual',                     0.48),
        (f'FiLMConvBlock  Conv1d({C}→{C})  +  GN  +  SiLU  +  FiLM  +  residual',                     0.39),
        (f'AxialAttentionBlock  MHA({C}, heads=4)  +  FiLM  +  residual',                              0.30),
        (f'FiLMConvBlock  (tail 1)',                                                                     0.21),
        (f'FiLMConvBlock  (tail 2)',                                                                     0.12),
        (f'Head:  Conv1d({C}→{dim}, k=1)  — raw velocity, no activation',                              0.03),
    ]

    box_w, box_h = 0.84, 0.058
    x0_box = 0.08

    for label, yc in arch_layers:
        ax_arch.add_patch(mpatches.FancyBboxPatch(
            (x0_box, yc - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.012',
            linewidth=1.2, edgecolor='steelblue', facecolor='white', zorder=3
        ))
        ax_arch.text(x0_box + box_w / 2, yc, label,
                     ha='center', va='center', color='black', fontsize=8.0, zorder=4)

    for i in range(len(arch_layers) - 1):
        y_top = arch_layers[i][1] - box_h / 2
        y_bot = arch_layers[i + 1][1] + box_h / 2
        ax_arch.annotate('', xy=(0.5, y_bot + 0.002), xytext=(0.5, y_top - 0.002),
                         arrowprops=dict(arrowstyle='->', color='grey', lw=1.2), zorder=5)

    ax_arch.text(0.5, -0.04,
                 f'cond = [t(1) | context({context_dim})]  injected via FiLM into every block',
                 ha='center', va='bottom', color='grey', fontsize=7.5, style='italic')

    plt.tight_layout()
    plt.savefig('flow_matching_result.png', dpi=150, bbox_inches='tight')
    plt.show()