import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. Domain and Target Distributions
# ==========================================

def sample_spherical_shell(batch_size, delta=0.05):
    """Samples uniformly from the spherical latent domain D_delta (3D shell)."""
    # Sample uniformly in volume: r ~ U(delta^3, 1)^(1/3)
    s = torch.rand(batch_size)
    r = (delta**3 + (1 - delta**3) * s)**(1/3)
    
    # Sample angles uniformly
    # phi ~ U(0, 2pi), cos_theta ~ U(-1, 1)
    phi = 2 * torch.pi * torch.rand(batch_size)
    cos_theta = 2 * torch.rand(batch_size) - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)
    
    x = r * sin_theta * torch.cos(phi)
    y = r * sin_theta * torch.sin(phi)
    z = r * cos_theta
    return torch.stack([x, y, z], dim=1)

def sample_target_gmm(batch_size):
    """Samples from a target spatial density (Two Gaussian clusters in 3D)."""
    modes = torch.tensor([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    choices = torch.randint(0, 2, (batch_size,))
    noise = torch.randn(batch_size, 3) * 0.15
    return modes[choices] + noise

# ==========================================
# 2. Velocity Field Network
# ==========================================

class VelocityNet(nn.Module):
    """MLP parametrizing the velocity field v_theta(s, y) in 3D."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Input: 1D time (s) + 3D coordinate (y) = 4
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3) # Output: 3D velocity
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
    """Standard Optimal Transport Conditional Flow Matching (3D)."""
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z0 = sample_spherical_shell(batch_size, delta)
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
# 5. Clean 3D Volumetric Lawnmower Path
# ==========================================

def generate_lawnmower_path_3d(pretrained_model, N_x=8, N_y=8, pts_per_z_line=20, delta=0.05):
    """
    Generates a continuous 3D up-and-down volumetric lawnmower.
    We sweep across X and Y within the disk, and go up and down in Z.
    """
    print("\nGenerating Clean 3D Volumetric Lawnmower Path...")
    
    max_radius = 0.95 
    x_grid = np.linspace(-max_radius, max_radius, N_x)
    
    raw_points = []
    going_up_z = True
    going_up_y = True
    
    for x in x_grid:
        # For a given X, find the max Y extent
        y_max_bound = np.sqrt(max_radius**2 - x**2)
        
        if going_up_y:
            y_vals = np.linspace(-y_max_bound, y_max_bound, N_y)
        else:
            y_vals = np.linspace(y_max_bound, -y_max_bound, N_y)
            
        for y in y_vals:
            # Max Z bounds for given (X, Y)
            z_max = np.sqrt(max(0, max_radius**2 - x**2 - y**2))
            
            if going_up_z:
                z_vals = np.linspace(-z_max, z_max, pts_per_z_line)
            else:
                z_vals = np.linspace(z_max, -z_max, pts_per_z_line)
                
            for z in z_vals:
                r = np.sqrt(x**2 + y**2 + z**2)
                # If the sweep crosses the tiny base station hole, route it along the edge
                if r < delta:
                    factor = delta / (r + 1e-6)
                    raw_points.append([x * factor, y * factor, z * factor])
                else:
                    raw_points.append([x, y, z])
                    
            going_up_z = not going_up_z
        going_up_y = not going_up_y
        
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
    
    print("Initializing 3D Model...")
    model = VelocityNet(hidden_dim=128)
    
    print("Training Infinite-Horizon Pushforward Map (Fast OT-CFM in 3D)...")
    train_cfm_fast(model, epochs=EPOCHS, batch_size=512, delta=DELTA)
    
    # ---------------- Latent Path Generation ----------------
    print("\n--- Map Trained! Generating 3D Flight Path ---")
    z_sves_path, x_sves_path = generate_lawnmower_path_3d(
        model, N_x=8, N_y=8, pts_per_z_line=20, delta=DELTA
    )
    
    # ---------------- Plotting Trajectories ----------------
    print("\nRendering 3D Results...")
    fig = plt.figure(figsize=(18, 6))
    
    # Panel 1: Target Reference
    ax1 = fig.add_subplot(131, projection='3d')
    target_pts = sample_target_gmm(1500)
    ax1.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], s=2, alpha=0.5, c='purple')
    ax1.set_title("Target Density (GMM 3D)")
    ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2); ax1.set_zlim(-1.2, 1.2)
    
    # Panel 2: The Latent Space (Clean Volumetric Lawnmower)
    ax2 = fig.add_subplot(132, projection='3d')
    z_np = z_sves_path.numpy()
    
    # Plot continuous sequential path lines
    ax2.plot(z_np[:, 0], z_np[:, 1], z_np[:, 2], 'b-', linewidth=1.2, alpha=0.6)
    ax2.scatter(z_np[:, 0], z_np[:, 1], z_np[:, 2], s=4, c='blue', zorder=5)
    
    # Plot outer wireframe sphere for reference
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u)*np.sin(v)
    y_sphere = np.sin(u)*np.sin(v)
    z_sphere = np.cos(v)
    ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color="black", alpha=0.1)
    
    ax2.set_title("Latent Path (3D Volumetric Lawnmower)")
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_zlim(-1.2, 1.2)
    
    # Panel 3: The Mapped Result (Real World Scan)
    ax3 = fig.add_subplot(133, projection='3d')
    x_np = x_sves_path.numpy()
    ax3.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], s=2, alpha=0.1, c='purple')
    
    # Plot the mapped path contour tracks
    ax3.plot(x_np[:, 0], x_np[:, 1], x_np[:, 2], 'orange', linewidth=1.5)
    ax3.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], s=4, c='red', zorder=5)
    ax3.set_title("Continuous Mapped Ergodic Sweep 3D")
    ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2); ax3.set_zlim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig('3d_visualization.png')
    print('Plot saved to 3d_visualization.png')

if __name__ == "__main__":
    main()
