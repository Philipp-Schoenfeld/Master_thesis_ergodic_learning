"""
Flow Matching: A Step-by-Step Implementation Tutorial

Flow matching is a generative model that learns a continuous vector field to 
transform a simple distribution (like Gaussian noise) into a complex data distribution 
(like the two-moons dataset) over time t, where t goes from 0 to 1.
"""

import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# =============================================================================
# STEP 1: DEFINING THE CONTINUOUS VECTOR FIELD (THE BRAIN)
# =============================================================================
class Flow(nn.Module):
    """
    This neural network acts as our velocity field. 
    Given a current spatial position (x_t) and a specific time (t), it predicts 
    the velocity vector needed to stay on the optimal path toward the target data.
    """
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        # We use an MLP (Multi-Layer Perceptron).
        # Input dimension is dim + 1 because we must process the 2D spatial 
        # coordinates AND the 1D time coordinate simultaneously.
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), 
            nn.ELU(), # ELU is smoother than ReLU, preventing jagged continuous paths
            nn.Linear(h, h), 
            nn.ELU(),
            nn.Linear(h, h), 
            nn.ELU(),
            # Output is strictly the spatial dimensions (the 2D velocity vector)
            nn.Linear(h, dim)
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        # torch.cat merges time and space into a single input vector.
        # dim=-1 ensures we concatenate along the feature dimension, preserving the batch size.
        return self.net(torch.cat((t, x_t), -1))

    # =========================================================================
    # STEP 2: THE ODE INTEGRATOR (THE ENGINE)
    # =========================================================================
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        """
        To move a point through space, we must integrate the learned velocity field 
        over time. This function uses the Midpoint Method (a 2nd-order ODE solver).
        """
        # Shape alignment: Ensure the time tensor matches the batch size of the spatial tensor
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        
        # Midpoint Integration Math:
        # 1. Calculate the time step (dt)
        dt = t_end - t_start
        
        # 2. Get the velocity at the starting position
        v_start = self(x_t, t_start)
        
        # 3. Take a temporary half-step to find the midpoint
        x_mid = x_t + v_start * (dt / 2)
        t_mid = t_start + (dt / 2)
        
        # 4. Get the velocity at the midpoint
        v_mid = self(x_mid, t_mid)
        
        # 5. Take the full step from the STARTING position using the MIDPOINT velocity
        x_next = x_t + v_mid * dt
        
        return x_next

# =============================================================================
# STEP 3: SIMULATION-FREE TRAINING LOOP
# =============================================================================
# We don't train by running the ODE solver (which is slow). Instead, we construct 
# perfectly straight mathematical lines between noise and data, and force the 
# network to memorize those specific velocity vectors.

flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

print("Starting training loop...")
for epoch in range(10000): 
    # 1. Sample real data (x_1) and standard Gaussian noise (x_0)
    x_1 = Tensor(make_moons(256, noise=0.05)[0])
    x_0 = torch.randn_like(x_1)
    
    # 2. Sample random time steps uniformly between 0 and 1
    t = torch.rand(len(x_1), 1)
    
    # 3. Interpolation: Where exactly should the point be at time t?
    # Equation: x_t = (1 - t) * x_0 + t * x_1
    x_t = (1 - t) * x_0 + t * x_1
    
    # 4. Target Velocity: The derivative of the interpolation equation
    # The velocity of a straight line between x_0 and x_1 is simply their difference.
    dx_t = x_1 - x_0
    
    # 5. Optimization
    optimizer.zero_grad()
    
    # Pass current state (x_t) and time (t) to the model. 
    # Calculate MSE loss between predicted velocity and the true straight-line velocity.
    predicted_velocity = flow(x_t, t)
    loss = loss_fn(predicted_velocity, dx_t)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:05d} | Loss: {loss.item():.4f}")

# =============================================================================
# STEP 4: INFERENCE AND SAMPLING
# =============================================================================
# Now that the vector field is trained, we drop pure noise into it at t=0 
# and integrate forward to t=1 to see if it shapes into the two-moons dataset.

print("Generating samples...")
x = torch.randn(300, 2) # Start with 300 points of pure noise
n_steps = 8             # Number of integration steps

# Set up the visualization plot
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)

# Create a timeline from 0.0 to 1.0
time_steps = torch.linspace(0, 1.0, n_steps + 1)

# Plot the initial noise state (t=0)
axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
axes[0].set_title(f't = {time_steps[0]:.2f}')
axes[0].set_xlim(-3.0, 3.0)
axes[0].set_ylim(-3.0, 3.0)

# Iteratively push the noise through the ODE integrator
with torch.no_grad(): # Disable gradients during inference to save memory
    for i in range(n_steps):
        # Update the positions x using our trained neural network and midpoint ODE solver
        x = flow.step(x, time_steps[i], time_steps[i + 1])
        
        # Plot the intermediate positions
        axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
        axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

plt.tight_layout()
plt.show()
