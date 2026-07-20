import torch
import torch.nn as nn

# ==========================================
# STEP 1: The Velocity Field (The Brain)
# ==========================================
class Flow(nn.Module):
    def __init__(self):
        # 1. Initialize the parent nn.Module class
        super().__init__()
        
        # 2. Define the Neural Network Architecture
        # nn.Sequential runs the data through these layers in order.
        self.net = nn.Sequential(
            # Input Layer: 3 inputs (X, Y, and Time) -> 64 hidden neurons
            nn.Linear(3, 64), 
            nn.ELU(), # Smooth activation function (better than ReLU for ODEs)
            
            # Hidden Layers
            nn.Linear(64, 64), 
            nn.ELU(),
            nn.Linear(64, 64), 
            nn.ELU(),
            
            # Output Layer: 64 hidden neurons -> 2 outputs (Velocity X, Velocity Y)
            nn.Linear(64, 2)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        The forward pass. PyTorch calls this automatically when you do: model(x_t, t)
        """
        # x_t is shape [batch_size, 2]
        # t is shape [batch_size, 1]
        
        # We must glue them together into a single matrix of shape [batch_size, 3]
        # dim=-1 means "glue them along the last dimension" (the features)
        combined_input = torch.cat((t, x_t), dim=-1)
        
        # Pass the combined matrix through the neural network and return the velocity
        return self.net(combined_input)

    # ==========================================
    # STEP 2: The ODE Integrator (The Engine)
    # ==========================================
    def midpoint_step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        """
        Pushes a batch of points forward in time using the Midpoint Method.
        """
        # --- Safety Check for Beginners ---
        # Later on, we might accidentally pass regular numbers (like 0.1) instead of Tensors.
        # This converts them to Tensors if necessary.
        if not isinstance(t_start, torch.Tensor):
            t_start = torch.tensor([t_start], dtype=torch.float32)
        if not isinstance(t_end, torch.Tensor):
            t_end = torch.tensor([t_end], dtype=torch.float32)
            
        # --- Tensor Reshaping ---
        # We need t_start to have the exact same number of rows as x_t.
        # .view(-1, 1) forces it to be a column vector.
        # .expand(...) duplicates the time value for every point in the batch.
        t_start = t_start.view(-1, 1).expand(x_t.shape[0], 1)
        t_end = t_end.view(-1, 1).expand(x_t.shape[0], 1)
        
        # Calculate the size of the time step
        dt = t_end - t_start
        
        # --- The Midpoint Math ---
        # 1. Ask the network for the velocity at the starting line
        v_start = self.forward(x_t, t_start)
        
        # 2. Take a temporary "half-step" forward to see where we land
        x_half = x_t + v_start * (dt / 2.0)
        t_half = t_start + (dt / 2.0)
        
        # 3. Ask the network for the velocity at that new half-way position
        v_half = self.forward(x_half, t_half)
        
        # 4. Take the real, full step from the STARTING position, 
        # but use the highly-accurate half-way velocity.
        x_next = x_t + v_half * dt
        
        return x_next

# --- Quick Test ---
# If you run this file, this block will execute to prove the shapes work.
if __name__ == "__main__":
    # Create an instance of our model
    model = Flow()
    
    # Create 5 fake data points (batch_size=5, coordinates=2)
    fake_data = torch.randn(5, 2)
    
    # Take a step from time 0.0 to 0.1
    # We don't need gradients for this test, so we use torch.no_grad() to save memory
    with torch.no_grad():
        new_positions = model.midpoint_step(fake_data, t_start=0.0, t_end=0.1)
        
    print("Original Positions:\n", fake_data)
    print("\nNew Positions after taking a step in time:\n", new_positions)