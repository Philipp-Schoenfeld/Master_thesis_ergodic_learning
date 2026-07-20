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


class CNNTrajectoryFlowNetwork(nn.Module):
    def __init__(self, dim=2, context_dim=2, hidden_channels=64):
        super().__init__()
        self.dim = dim
        in_channels = dim + 1 + context_dim
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(), # Vielleicht eine andere versuchen  
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

    model = CNNTrajectoryFlowNetwork(dim=dim, context_dim=context_dim).to(device)
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
            x1_batch      = x1_batch.to(device)
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

    plt.figure(figsize=(8, 8))
    plt.plot(final_trajs_np[0, :, 0], final_trajs_np[0, :, 1], marker='o', linestyle='-', color='blue')
    plt.scatter([test_context[0].item()], [test_context[1].item()], color='red', s=100, label='Conditioned Start (x0)', zorder=5)
    plt.title(f"Conditioned B-Spline Generation (Start: {test_context.tolist()})")
    plt.legend()
    plt.grid(True)
    plt.show()