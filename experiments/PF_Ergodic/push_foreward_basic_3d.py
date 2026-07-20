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
    """
    Standard Optimal Transport Conditional Flow Matching.
    Trains extremely fast because we are not running ODE solvers inside the loop.
    """
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z0 = sample_annulus(batch_size, delta)
        x1 = sample_target_gmm(batch_size)
        
        # Optimal Transport Coupling
        C = torch.cdist(z0, x1)**2 
        row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
        x1_coupled = x1[col_ind] 
        
        # Flow Matching MSE Loss
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
# 5. Latent Stein Variational Ergodic Search
# ==========================================

def optimize_latent_sves_path(pretrained_model, N_waypoints=150, delta=0.05, steps=200):
    """
    Latent Stein Variational Ergodic Search (SVES) using the Windshield Wiper.
    Generates a natively continuous, sweeping flight path.
    """
    print("\nStarting Latent SVES with Windshield Wiper Initialization...")
    
    # 1. INITIALIZATION: The Polar Sweep (Windshield Wiper)
    t = torch.linspace(0, 1, N_waypoints)
    
    # Radius slowly expands from inner delta to outer boundary 1
    r_vals = torch.sqrt(delta**2 + (1 - delta**2) * t)
    
    # Angle swipes back and forth (Triangle wave between 0 and 2*pi)
    sweeps = 4.0  # The drone will do 4 full back-and-forth passes
    triangle_wave = 1.0 - torch.abs((t * sweeps * 2.0) % 2.0 - 1.0)
    theta_vals = triangle_wave * 2.0 * torch.pi
    
    z_init = torch.stack([r_vals * torch.cos(theta_vals), r_vals * torch.sin(theta_vals)], dim=1)
    
    # Make it learnable for SVGD
    z_pts = z_init.clone().detach()
    z_pts.requires_grad = True
    
    # Gentle SGD to let the points repel and settle perfectly
    optimizer = optim.SGD([z_pts], lr=0.1, momentum=0.8)
    h = 0.2 # SVGD Kernel Bandwidth
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # 2. SVGD Repulsion (The "Magnetic" Field)
        diffs = z_pts.unsqueeze(1) - z_pts.unsqueeze(0) 
        sq_dists = torch.sum(diffs**2, dim=-1)           
        K_xx = torch.exp(-sq_dists / h)                  
        
        repulsive_force = - (2.0 / h) * diffs * K_xx.unsqueeze(-1)
        svgd_update = repulsive_force.mean(dim=1)        
        
        # Maximize distance
        fake_loss = torch.sum(-svgd_update.detach() * z_pts)
        fake_loss.backward()
        
        optimizer.step()
        
        # 3. Boundary Condition Clamp
        with torch.no_grad():
            r = torch.norm(z_pts, dim=1)
            r_clamped = torch.clamp(r, min=delta, max=1.0)
            z_pts.data = z_pts.data * (r_clamped / r).unsqueeze(1)
            
        if step % 50 == 0:
            print(f"SVES Step {step:03d} | Max Repulsive Force: {svgd_update.abs().max().item():.4f}")

    # 4. Skip TSP: Because we initialized with a continuous sweep, the array 
    # index is ALREADY the perfect continuous flight path.
    z_path = z_pts.detach()
    
    # 5. Push the optimized, continuous path through the frozen Neural Network map
    print("Pushing swiping SVES path through the learned map...")
    x_path = pushforward_trajectory(pretrained_model, z_path, steps=20)
    
    return z_path, x_path

# ==========================================
# 6. Main Execution & Visualization
# ==========================================

def main():
    DELTA = 0.05
    EPOCHS = 1500
    N_WAYPOINTS = 150
    
    print("Initializing Model...")
    model = VelocityNet(hidden_dim=128)
    
    print("Training Infinite-Horizon Pushforward Map (Fast OT-CFM)...")
    train_cfm_fast(model, epochs=EPOCHS, batch_size=512, delta=DELTA)
    
    # ---------------- Latent SVES Optimization ----------------
    print("\n--- Map Trained! Running Latent SVES ---")
    z_sves_path, x_sves_path = optimize_latent_sves_path(
        model, N_waypoints=N_WAYPOINTS, delta=DELTA, steps=200
    )
    
    # ---------------- Plotting Trajectories ----------------
    print("\nRendering SVES Results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Target Reference
    target_pts = sample_target_gmm(1500)
    axes[0].scatter(target_pts[:, 0], target_pts[:, 1], s=2, alpha=0.5, c='purple')
    axes[0].set_title("Target Density (GMM)")
    axes[0].set_xlim(-1.2, 1.2); axes[0].set_ylim(-1.2, 1.2)
    
    # Panel 2: The Latent Space (The Windshield Wiper SVGD Path)
    z_np = z_sves_path.numpy()
    
    # Plot the sequential path
    axes[1].plot(z_np[:, 0], z_np[:, 1], 'b-', linewidth=1.5, alpha=0.6)
    # Scatter the optimized waypoints to show distribution
    axes[1].scatter(z_np[:, 0], z_np[:, 1], s=12, c='blue', zorder=5)
    
    circle = plt.Circle((0, 0), DELTA, color='black', fill=False)
    outer = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--')
    axes[1].add_patch(circle)
    axes[1].add_patch(outer)
    axes[1].set_title(f"Latent SVES (Windshield Wiper)")
    axes[1].set_xlim(-1.2, 1.2); axes[1].set_ylim(-1.2, 1.2)
    
    # Panel 3: The Mapped Result (The Real World Flight)
    x_np = x_sves_path.numpy()
    axes[2].scatter(target_pts[:, 0], target_pts[:, 1], s=2, alpha=0.1, c='purple')
    
    # Plot the mapped path
    axes[2].plot(x_np[:, 0], x_np[:, 1], 'orange', linewidth=2.0)
    # Scatter the mapped waypoints
    axes[2].scatter(x_np[:, 0], x_np[:, 1], s=12, c='red', zorder=5)
    axes[2].set_title("Continuous Sweeping Real-World Path")
    axes[2].set_xlim(-1.2, 1.2); axes[2].set_ylim(-1.2, 1.2)
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()