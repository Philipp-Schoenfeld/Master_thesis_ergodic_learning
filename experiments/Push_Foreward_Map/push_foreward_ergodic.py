import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. Domain and Target Distributions
# ==========================================

def sample_annulus(batch_size, delta=0.05):
    """Samples uniformly from the annular latent domain D_delta."""
    s = torch.rand(batch_size)
    r = torch.sqrt(delta**2 + (1 - delta**2) * s)
    theta = 2 * torch.pi * torch.rand(batch_size)
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)

def sample_target_gmm(batch_size):
    """Samples from a target spatial density (Two Gaussian clusters)."""
    modes = torch.tensor([[-0.4, 0.0], [0.4, 0.0]])
    choices = torch.randint(0, 2, (batch_size,))
    noise = torch.randn(batch_size, 2) * 0.15
    return modes[choices] + noise

# ==========================================
# 2. Velocity Field Network
# ==========================================

class VelocityNet(nn.Module):
    """MLP parametrizing the velocity field v_theta(s, y)."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # ZERO-INITIALIZATION: Start with an identity map (zero velocity)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, y):
        if s.dim() == 1:
            s = s.unsqueeze(1)
        sy = torch.cat([s, y], dim=1)
        return self.net(sy)

# ==========================================
# 3. Fast OT-CFM Training Routine
# ==========================================

def train_cfm_fast(model, epochs=1500, batch_size=512, delta=0.05):
    """Standard Optimal Transport Conditional Flow Matching."""
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z0 = sample_annulus(batch_size, delta)
        x1 = sample_target_gmm(batch_size)
        
        C = torch.cdist(z0, x1)**2 
        row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
        x1_coupled = x1[col_ind] 
        
        s = torch.rand(batch_size, 1)
        y_s = (1 - s) * z0 + s * x1_coupled
        target_velocity = x1_coupled - z0
        
        pred_velocity = model(s, y_s)
        loss = torch.mean((pred_velocity - target_velocity)**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch:04d} | Fast CFM Loss: {loss.item():.4f}")

# ==========================================
# 4. Trajectory Generation & ODE Solver
# ==========================================

def rk4_step(model, s, y, ds):
    """Runge-Kutta 4th order numerical integration step."""
    k1 = model(s, y)
    k2 = model(s + ds/2, y + k1 * ds/2)
    k3 = model(s + ds/2, y + k2 * ds/2)
    k4 = model(s + ds, y + k3 * ds)
    return y + (ds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@torch.no_grad()
def pushforward_trajectory(model, z_traj, steps=20):
    """Warps a latent path into the target environment."""
    model.eval()
    y = z_traj.clone()
    ds = 1.0 / steps
    
    for i in range(steps):
        s = torch.full((y.size(0), 1), i * ds)
        y = rk4_step(model, s, y, ds)
        
    return y

# ==========================================
# 5. Clean Circular Lawnmower Path
# ==========================================

def generate_lawnmower_path(pretrained_model, N_lines=16, pts_per_line=20, delta=0.05):
    """
    Generates a True Circular Up-and-Down Lawnmower.
    We remove SVGD repulsion so the grid remains perfectly straight and 
    points don't get smashed against the outer walls.
    """
    print("\nGenerating Clean Circular Lawnmower Path...")
    
    # Inset slightly from 1.0 so the drone doesn't ride the absolute boundary wall
    max_radius = 0.95 
    x_grid = np.linspace(-max_radius, max_radius, N_lines)
    
    raw_points = []
    going_up = True
    
    for x in x_grid:
        # Calculate the maximum vertical boundary of the circle at this X
        y_max = np.sqrt(max_radius**2 - x**2)
        
        # Generate the vertical sweep (up or down)
        if going_up:
            y_vals = np.linspace(-y_max, y_max, pts_per_line)
        else:
            y_vals = np.linspace(y_max, -y_max, pts_per_line)
            
        for y in y_vals:
            r = np.sqrt(x**2 + y**2)
            # If the sweep crosses the tiny base station hole, route it along the edge
            if r < delta:
                factor = delta / (r + 1e-6)
                raw_points.append([x * factor, y * factor])
            else:
                raw_points.append([x, y])
                
        going_up = not going_up
        
    z_path = torch.tensor(np.array(raw_points), dtype=torch.float32)
    
    # Push continuous lawnmower path through the map
    print("Pushing continuous lawnmower path through the learned map...")
    x_path = pushforward_trajectory(pretrained_model, z_path, steps=20)
    
    return z_path, x_path

# ==========================================
# 6. Main Execution & Visualization
# ==========================================

def main():
    DELTA = 0.05
    EPOCHS = 1500
    
    print("Initializing Model...")
    model = VelocityNet(hidden_dim=128)
    
    print("Training Infinite-Horizon Pushforward Map (Fast OT-CFM)...")
    train_cfm_fast(model, epochs=EPOCHS, batch_size=512, delta=DELTA)
    
    # ---------------- Latent Path Generation ----------------
    print("\n--- Map Trained! Generating Flight Path ---")
    z_sves_path, x_sves_path = generate_lawnmower_path(
        model, N_lines=16, pts_per_line=20, delta=DELTA
    )
    
    # ---------------- Plotting Trajectories ----------------
    print("\nRendering Results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Target Reference
    target_pts = sample_target_gmm(1500)
    axes[0].scatter(target_pts[:, 0], target_pts[:, 1], s=2, alpha=0.5, c='purple')
    axes[0].set_title("Target Density (GMM)")
    axes[0].set_xlim(-1.2, 1.2); axes[0].set_ylim(-1.2, 1.2)
    
    # Panel 2: The Latent Space (Clean Circular Lawnmower)
    z_np = z_sves_path.numpy()
    
    # Plot continuous sequential path lines
    axes[1].plot(z_np[:, 0], z_np[:, 1], 'b-', linewidth=1.2, alpha=0.6)
    axes[1].scatter(z_np[:, 0], z_np[:, 1], s=8, c='blue', zorder=5)
    
    circle = plt.Circle((0, 0), DELTA, color='black', fill=False)
    outer = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--')
    axes[1].add_patch(circle)
    axes[1].add_patch(outer)
    axes[1].set_title("Latent Path (Clean Lawnmower)")
    axes[1].set_xlim(-1.2, 1.2); axes[1].set_ylim(-1.2, 1.2)
    
    # Panel 3: The Mapped Result (Real World Scan)
    x_np = x_sves_path.numpy()
    axes[2].scatter(target_pts[:, 0], target_pts[:, 1], s=2, alpha=0.1, c='purple')
    
    # Plot the mapped path contour tracks
    axes[2].plot(x_np[:, 0], x_np[:, 1], 'orange', linewidth=1.5)
    axes[2].scatter(x_np[:, 0], x_np[:, 1], s=8, c='red', zorder=5)
    axes[2].set_title("Continuous Mapped Ergodic Sweep")
    axes[2].set_xlim(-1.2, 1.2); axes[2].set_ylim(-1.2, 1.2)
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()