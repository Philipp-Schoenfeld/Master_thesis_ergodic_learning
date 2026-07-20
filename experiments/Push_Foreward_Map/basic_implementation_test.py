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
    """
    Samples uniformly from the annular latent domain D_delta.
    Paper: r(s) = sqrt(delta^2 + (1-delta^2)*s) where s ~ U[0,1] 
    and theta ~ U[0, 2pi)[cite: 1667, 1669].
    """
    s = torch.rand(batch_size)
    r = torch.sqrt(delta**2 + (1 - delta**2) * s)
    theta = 2 * torch.pi * torch.rand(batch_size)
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)

def sample_target_gmm(batch_size):
    """
    Samples from a target spatial density. 
    Using a 2D Gaussian Mixture Model similar to Experiment 1[cite: 2483].
    """
    modes = torch.tensor([[-0.4, 0.0], [0.4, 0.0]])
    choices = torch.randint(0, 2, (batch_size,))
    noise = torch.randn(batch_size, 2) * 0.15
    return modes[choices] + noise

# ==========================================
# 2. Velocity Field Network
# ==========================================

class VelocityNet(nn.Module):
    """
    MLP parametrizing the velocity field v_theta(s, y).
    Concatenates flow time 's' with spatial coordinates 'y'[cite: 2448].
    """
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

    def forward(self, s, y):
        # Ensure s is shape (batch_size, 1) to match y (batch_size, 2)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        sy = torch.cat([s, y], dim=1)
        return self.net(sy)

# ==========================================
# 3. OT-CFM Training Routine
# ==========================================

def train_cfm(model, epochs=1500, batch_size=512, delta=0.05):
    """
    Trains the Conditional Flow Matching model with Optimal Transport coupling.
    Minimizes the loss: E || v_theta(s, y_s) - (x_1 - z_0) ||^2[cite: 1696].
    """
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Sample source (z_0) and target (x_1)
        z0 = sample_annulus(batch_size, delta)
        x1 = sample_target_gmm(batch_size)
        
        # 2. Optimal Transport Coupling (Mini-batch)
        # Calculate pairwise squared distances
        C = torch.cdist(z0, x1)**2 
        C_np = C.detach().cpu().numpy()
        
        # Exact linear sum assignment (Hungarian algorithm) to approximate OT
        row_ind, col_ind = linear_sum_assignment(C_np)
        x1_coupled = x1[col_ind] # Rearrange target to minimize transport cost
        
        # 3. Sample flow time s ~ U[0, 1]
        s = torch.rand(batch_size, 1)
        
        # 4. Interpolate path y_s and compute target velocity
        y_s = (1 - s) * z0 + s * x1_coupled
        target_velocity = x1_coupled - z0
        
        # 5. Predict velocity and compute MSE loss
        pred_velocity = model(s, y_s)
        loss = torch.mean((pred_velocity - target_velocity)**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch:04d} | CFM Loss: {loss.item():.4f}")

# ==========================================
# 4. Trajectory Generation & Inference
# ==========================================

def rk4_step(model, s, y, ds):
    """Runge-Kutta 4th order numerical integration step[cite: 2453]."""
    k1 = model(s, y)
    k2 = model(s + ds/2, y + k1 * ds/2)
    k3 = model(s + ds/2, y + k2 * ds/2)
    k4 = model(s + ds, y + k3 * ds)
    return y + (ds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@torch.no_grad()
def generate_latent_trajectory(K_cycles, delta=0.05, pts_per_half_cycle=50):
    """
    Generates the analytical, ergodic latent trajectory on D_delta.
    Consists of radial back-and-forth traversals[cite: 1669].
    """
    traj = []
    for _ in range(K_cycles):
        # Draw random heading for the cycle
        theta = np.random.uniform(0, 2*np.pi)
        
        # Outward leg
        s_vals = np.linspace(0, 1, pts_per_half_cycle)
        r_vals = np.sqrt(delta**2 + (1 - delta**2) * s_vals)
        x_out = r_vals * np.cos(theta)
        y_out = r_vals * np.sin(theta)
        leg_out = np.stack([x_out, y_out], axis=1)
        
        # Return leg (reverse path)
        leg_in = leg_out[::-1]
        
        traj.append(leg_out)
        traj.append(leg_in)
        
    return torch.tensor(np.concatenate(traj), dtype=torch.float32)

@torch.no_grad()
def pushforward_trajectory(model, z_traj, steps=20):
    """
    Transforms the latent trajectory into the target domain by integrating
    the learned velocity field from s=0 to s=1[cite: 1695].
    """
    model.eval()
    y = z_traj.clone()
    ds = 1.0 / steps
    
    for i in range(steps):
        s = torch.full((y.size(0), 1), i * ds)
        y = rk4_step(model, s, y, ds)
        
    return y

# ==========================================
# 5. Main Execution & Visualization
# ==========================================

def main():
    DELTA = 0.05
    K_CYCLES = 50
    
    print("Initializing Model...")
    model = VelocityNet(hidden_dim=128)
    
    print("Training OT-CFM Pushforward Map...")
    train_cfm(model, epochs=1500, batch_size=512, delta=DELTA)
    
    print("Generating Latent Trajectory...")
    z_traj = generate_latent_trajectory(K_CYCLES, delta=DELTA)
    
    print("Applying Learned Pushforward Map...")
    x_traj = pushforward_trajectory(model, z_traj, steps=30)
    
    # ---------------- Plotting ----------------
    print("Rendering Visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Target Reference
    target_pts = sample_target_gmm(3000).numpy()
    axes[0].scatter(target_pts[:, 0], target_pts[:, 1], s=2, alpha=0.5, c='purple')
    axes[0].set_title("Target Distribution")
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    
    # Latent Trajectory
    z_np = z_traj.numpy()
    axes[1].plot(z_np[:, 0], z_np[:, 1], alpha=0.6, linewidth=0.5)
    # Draw inner hole constraint
    circle = plt.Circle((0, 0), DELTA, color='black', fill=False)
    axes[1].add_patch(circle)
    axes[1].set_title("initialization Trajectory")
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    
    # Mapped Trajectory
    x_np = x_traj.numpy()
    axes[2].plot(x_np[:, 0], x_np[:, 1], alpha=0.6, linewidth=0.5, color='orange')
    axes[2].set_title("Mapped Ergodic Trajectory")
    axes[2].set_xlim(-1.2, 1.2)
    axes[2].set_ylim(-1.2, 1.2)
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()