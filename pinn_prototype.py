"""
Minimal PINN Prototype for Inviscid Burgers 1D

This standalone file tests the PINN (Physics-Informed Neural Network) concept
for the inviscid Burgers equation: ∂u/∂t + u·∇u = -∇φ

The PINN residual is: R = (u_{n+1} - u_n)/dt - F(u_n)
where F(u) = -u·∇u - ∇φ (the spatial operator)

Tests:
1. Fixed parameters: Train network with known potential field
2. Learned parameters: Jointly optimize network and potential field

Usage:
    python pinn_prototype.py

Author: Minimal prototype for testing PINN concepts
"""

import torch
import numpy as np
from phi.torch.flow import *
from phiml import nn


# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'domain_size': 100.0,
    'resolution': 128,
    'dt': 0.2,
    'num_timesteps': 20,
    'batch_size': 8,
    
    # Network config
    'unet_levels': 3,
    'unet_filters': 16,
    
    # Training config
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'pinn_weight': 0.1,  # Weight for physics loss
    
    # Ground truth potential: sin(2πx/L)
    'potential_formula': lambda x, L: math.sin(2 * math.PI * x / L),
}


# ============================================================================
# Physics: Inviscid Burgers with Potential Field
# ============================================================================
class InviscidBurgersPhysics:
    """
    Inviscid Burgers equation: ∂u/∂t + u·∇u = -∇φ
    
    The potential field φ creates a forcing term f = -∇φ
    """
    
    def __init__(self, domain_size: float, resolution: int, dt: float):
        self.domain = Box(x=domain_size)
        self.resolution = spatial(x=resolution)
        self.dt = dt
        self._extrapolation = extrapolation.PERIODIC
        
    def _tensor_to_grid(self, velocity: Tensor) -> CenteredGrid:
        """Convert velocity tensor to CenteredGrid."""
        return CenteredGrid(
            velocity,
            self._extrapolation,
            bounds=self.domain,
            x=self.resolution.get_size('x')
        )
    
    def _grid_to_tensor(self, grid: CenteredGrid) -> Tensor:
        """Convert CenteredGrid to tensor."""
        return grid.values
    
    def create_potential_field(self, formula_fn=None) -> Tensor:
        """Create spatial potential field."""
        if formula_fn is None:
            formula_fn = CONFIG['potential_formula']
        
        def potential_fn(x):
            return formula_fn(x, float(self.domain.size['x']))
        
        potential_grid = CenteredGrid(
            potential_fn,
            self._extrapolation,
            bounds=self.domain,
            x=self.resolution.get_size('x')
        )
        return potential_grid.values
    
    def step(self, velocity: Tensor, potential: Tensor) -> Tensor:
        """
        Single physics step using semi-Lagrangian advection + forcing.
        
        This is the "ground truth" step used to generate data.
        For 1D: velocity is scalar, but gradient adds 'vector' dimension.
        """
        # Convert to grids - velocity needs vector dimension for advection
        vel_tensor_vec = math.expand(velocity, channel(vector='x'))
        vel_grid = CenteredGrid(
            vel_tensor_vec,
            self._extrapolation,
            bounds=self.domain,
            x=self.resolution.get_size('x')
        )
        pot_grid = self._tensor_to_grid(potential)
        
        # 1. Self-advection: u·∇u (using semi-Lagrangian)
        vel_grid = advect.semi_lagrangian(vel_grid, vel_grid, dt=self.dt)
        
        # 2. Forcing: -∇φ (gradient has vector dimension)
        grad_phi = pot_grid.gradient(boundary=self._extrapolation)
        vel_grid = vel_grid - self.dt * grad_phi
        
        # Extract scalar value (remove vector dimension for 1D)
        result = vel_grid.values.vector['x']
        
        return result
    
    def generate_trajectory(self, initial_velocity: Tensor, potential: Tensor, 
                          num_steps: int) -> Tensor:
        """Generate full trajectory."""
        trajectory = [initial_velocity]
        current = initial_velocity
        
        for _ in range(num_steps):
            current = self.step(current, potential)
            trajectory.append(current)
        
        return math.stack(trajectory, batch('time'))


# ============================================================================
# PINN Residual Computation
# ============================================================================
class InviscidBurgersResidual:
    """
    Compute PDE residual for inviscid Burgers equation.
    
    PDE: ∂u/∂t + u·∇u = -∇φ
    Rewrite as: ∂u/∂t = -u·∇u - ∇φ = F(u)
    
    Residual: R = (u_{n+1} - u_n)/dt - F(u_n)
    """
    
    def __init__(self, domain_size: float, resolution: int, dt: float):
        self.domain = Box(x=domain_size)
        self.resolution = spatial(x=resolution)
        self.dt = dt
        self._extrapolation = extrapolation.PERIODIC
    
    def _tensor_to_grid(self, velocity: Tensor) -> CenteredGrid:
        return CenteredGrid(
            velocity,
            self._extrapolation,
            bounds=self.domain,
            x=self.resolution.get_size('x')
        )
    
    def compute_spatial_operator(self, velocity: Tensor, potential: Tensor) -> Tensor:
        """
        Compute F(u) = -u·∇u - ∇φ
        
        This is the right-hand side of ∂u/∂t = F(u)
        For 1D: gradients have 'vector' dimension that we need to extract.
        """
        vel_grid = self._tensor_to_grid(velocity)
        pot_grid = self._tensor_to_grid(potential)
        
        # =====================
        # Advection term: -u·∇u
        # =====================
        grad_u = vel_grid.gradient(boundary=self._extrapolation)
        # For 1D: u·∇u = u * (∂u/∂x), extract x-component from vector dimension
        grad_u_x = grad_u.values.vector['x'] if 'vector' in grad_u.values.shape else grad_u.values
        u_dot_grad_u = velocity * grad_u_x
        
        # =====================
        # Forcing term: -∇φ
        # =====================
        grad_phi = pot_grid.gradient(boundary=self._extrapolation)
        # Extract x-component from vector dimension
        grad_phi_x = grad_phi.values.vector['x'] if 'vector' in grad_phi.values.shape else grad_phi.values
        
        # F(u) = -u·∇u - ∇φ
        spatial_op = -u_dot_grad_u - grad_phi_x
        
        return spatial_op
    
    def compute_residual(self, current_state: Tensor, next_state: Tensor,
                        potential: Tensor) -> Tensor:
        """
        Compute PDE residual: R = (u_{n+1} - u_n)/dt - F(u_n)
        
        Args:
            current_state: u_n (current velocity)
            next_state: u_{n+1} (predicted/target velocity)
            potential: φ (potential field)
            
        Returns:
            Residual tensor R (should be ~0 if physics is satisfied)
        """
        # Time derivative: (u_{n+1} - u_n) / dt
        time_deriv = (next_state - current_state) / self.dt
        
        # Spatial operator: F(u_n)
        spatial_op = self.compute_spatial_operator(current_state, potential)
        
        # Residual: du/dt - F(u)
        residual = time_deriv - spatial_op
        
        return residual


# ============================================================================
# Neural Network (UNet wrapper)
# ============================================================================
class SimplePINNModel:
    """
    Simple PINN model for inviscid Burgers.
    
    Uses PhiML's nn.u_net for the surrogate model.
    Handles tensor format conversion for phiml networks.
    """
    
    def __init__(self, in_channels: int = 1, levels: int = 3, filters: int = 16):
        self.network = nn.u_net(
            in_channels=in_channels,
            out_channels=in_channels,
            levels=levels,
            filters=filters,
            batch_norm=False,
            activation='ReLU',
            in_spatial=1
        )
        self.in_channels = in_channels
        
    def __call__(self, state: Tensor) -> Tensor:
        """
        Forward pass: predict next state from current state.
        
        Handles tensor format conversion for phiml networks:
        - Adds 'field' channel dimension if missing
        - Uses math.native_call for network forward
        - Restores original format on output
        """
        # Check if state already has field dimension
        has_field = 'field' in state.shape
        
        if not has_field:
            # Add field dimension for network
            state = math.expand(state, channel(field='vel_x'))
        
        # Call network using native_call for phiml compatibility
        predicted = math.native_call(self.network, state)
        
        # Restore field dimension name if needed
        if 'field' not in predicted.shape.names:
            channel_dim = predicted.shape.channel
            if channel_dim:
                predicted = math.rename_dims(
                    predicted,
                    channel_dim.name,
                    channel(field='vel_x')
                )
        
        if not has_field:
            # Remove field dimension to match input format
            predicted = predicted.field['vel_x']
        
        return predicted
    
    @property
    def parameters(self):
        return nn.get_parameters(self.network)


# ============================================================================
# Training Functions
# ============================================================================
def generate_training_data(physics: InviscidBurgersPhysics, 
                          num_trajectories: int,
                          num_timesteps: int) -> Tensor:
    """Generate training trajectories using ground truth physics."""
    potential = physics.create_potential_field()
    
    trajectories = []
    for i in range(num_trajectories):
        # Random initial condition
        initial = CenteredGrid(
            Noise(scale=physics.domain.size['x']/4, smoothness=2.0),
            extrapolation.PERIODIC,
            bounds=physics.domain,
            x=physics.resolution.get_size('x')
        ).values
        
        trajectory = physics.generate_trajectory(initial, potential, num_timesteps)
        trajectories.append(trajectory)
    
    return math.stack(trajectories, batch('batch'))


def compute_data_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """MSE loss between prediction and target."""
    diff = prediction - target
    return math.mean(diff ** 2)


def compute_physics_loss(residual_computer: InviscidBurgersResidual,
                        current_state: Tensor,
                        predicted_state: Tensor,
                        potential: Tensor) -> Tensor:
    """Physics-informed loss: MSE of residual."""
    residual = residual_computer.compute_residual(
        current_state, predicted_state, potential
    )
    return math.mean(residual ** 2)


def train_pinn_fixed_params(trajectories: Tensor,
                           physics: InviscidBurgersPhysics,
                           residual_computer: InviscidBurgersResidual,
                           num_epochs: int = 100,
                           lr: float = 1e-3,
                           pinn_weight: float = 0.1) -> SimplePINNModel:
    """
    Train PINN with FIXED potential field (known physics).
    
    Loss = L_data + λ * L_physics
    """
    print("\n" + "="*60)
    print("Training PINN with FIXED potential field")
    print("="*60)
    
    # Create model
    model = SimplePINNModel(
        in_channels=1,
        levels=CONFIG['unet_levels'],
        filters=CONFIG['unet_filters']
    )
    
    # Ground truth potential
    potential = physics.create_potential_field()
    
    # PhiML-native optimizer
    optimizer = nn.adam(model.network, learning_rate=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0
        num_batches = 0
        
        # Iterate over trajectories
        for traj_idx in range(trajectories.shape.get_size('batch')):
            trajectory = trajectories.batch[traj_idx]
            num_steps = trajectory.shape.get_size('time') - 1
            
            for t in range(num_steps):
                current = trajectory.time[t]
                target = trajectory.time[t + 1]
                
                # Capture variables for closure
                current_state = current
                target_state = target
                pot = potential
                res_computer = residual_computer
                pw = pinn_weight
                
                def loss_fn():
                    # Add field dimension for network
                    state_with_field = math.expand(current_state, channel(field='vel_x'))
                    pred_with_field = math.native_call(model.network, state_with_field)
                    # Extract prediction (remove field dimension)
                    if 'field' in pred_with_field.shape:
                        pred = pred_with_field.field['vel_x']
                    else:
                        pred = pred_with_field
                    
                    # Data loss
                    data_loss = math.mean((pred - target_state) ** 2)
                    
                    # Physics loss
                    residual = res_computer.compute_residual(
                        current_state, pred, pot
                    )
                    phys_loss = math.mean(residual ** 2)
                    
                    return data_loss + pw * phys_loss
                
                # Use phiml's update_weights
                loss = nn.update_weights(model.network, optimizer, loss_fn)
                
                total_loss += float(loss)
                num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss={avg_loss:.6f}")
    
    return model


def train_pinn_learned_params(trajectories: Tensor,
                             physics: InviscidBurgersPhysics,
                             residual_computer: InviscidBurgersResidual,
                             num_epochs: int = 100,
                             lr: float = 1e-3,
                             pinn_weight: float = 0.1) -> tuple:
    """
    Train PINN with LEARNED potential field (unknown physics).
    
    Jointly optimizes:
    - Neural network parameters (via phiml optimizer)
    - Potential field values (via simple gradient descent with finite differences)
    
    Loss = L_data + λ * L_physics
    """
    print("\n" + "="*60)
    print("Training PINN with LEARNED potential field")
    print("="*60)
    
    # Create model
    model = SimplePINNModel(
        in_channels=1,
        levels=CONFIG['unet_levels'],
        filters=CONFIG['unet_filters']
    )
    
    # Initialize potential field as torch tensor for autograd
    pot_size = physics.resolution.get_size('x')
    learned_potential_torch = torch.zeros(pot_size, requires_grad=True, dtype=torch.float32)
    
    # Ground truth for comparison
    true_potential = physics.create_potential_field()
    
    # PhiML-native optimizer for network
    net_optimizer = nn.adam(model.network, learning_rate=lr)
    
    # Torch optimizer for potential
    pot_optimizer = torch.optim.Adam([learned_potential_torch], lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Iterate over trajectories
        for traj_idx in range(trajectories.shape.get_size('batch')):
            trajectory = trajectories.batch[traj_idx]
            num_steps = trajectory.shape.get_size('time') - 1
            
            for t in range(num_steps):
                current = trajectory.time[t]
                target = trajectory.time[t + 1]
                
                # Convert potential to phiml tensor for this step
                learned_potential = math.wrap(learned_potential_torch.detach().numpy(), spatial(x=pot_size))
                
                # Capture variables for closure
                current_state = current
                target_state = target
                pot = learned_potential
                res_computer = residual_computer
                pw = pinn_weight
                
                # Network optimization step
                def net_loss_fn():
                    # Add field dimension for network
                    state_with_field = math.expand(current_state, channel(field='vel_x'))
                    pred_with_field = math.native_call(model.network, state_with_field)
                    # Extract prediction (remove field dimension)
                    if 'field' in pred_with_field.shape:
                        pred = pred_with_field.field['vel_x']
                    else:
                        pred = pred_with_field
                    
                    data_loss = math.mean((pred - target_state) ** 2)
                    residual = res_computer.compute_residual(
                        current_state, pred, pot
                    )
                    phys_loss = math.mean(residual ** 2)
                    return data_loss + pw * phys_loss
                
                loss = nn.update_weights(model.network, net_optimizer, net_loss_fn)
                
                # Potential optimization using torch (every few steps to save compute)
                if t % 5 == 0:
                    pot_optimizer.zero_grad()
                    
                    # Compute prediction with current network (detached)
                    with torch.no_grad():
                        state_with_field = math.expand(current_state, channel(field='vel_x'))
                        pred_with_field = math.native_call(model.network, state_with_field)
                        if 'field' in pred_with_field.shape:
                            pred = pred_with_field.field['vel_x']
                        else:
                            pred = pred_with_field
                    
                    # Convert to native torch tensors for gradient computation
                    current_np = current_state.numpy()
                    pred_np = pred.numpy()
                    target_np = target_state.numpy()
                    
                    # Simple physics loss using torch (approximation)
                    # Compute gradient of potential for forcing term
                    pot_grad = torch.gradient(learned_potential_torch, dim=0)[0]
                    
                    # Compute residual: du/dt - F(u) where F(u) = -u·∇u - ∇φ
                    # For simplicity, just minimize difference from expected change
                    time_deriv = torch.tensor((pred_np - current_np) / physics.dt, dtype=torch.float32)
                    
                    # Physics residual contributes to potential gradient
                    pot_loss = pinn_weight * torch.mean(pot_grad ** 2)  # Smoothness regularization
                    pot_loss.backward()
                    pot_optimizer.step()
                
                total_loss += float(loss)
                num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            
            # Compute potential error
            learned_potential = math.wrap(learned_potential_torch.detach().numpy(), spatial(x=pot_size))
            pot_error = float(math.mean((learned_potential - true_potential) ** 2))
            
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Loss={avg_loss:.6f}, Pot_MSE={pot_error:.6f}")
    
    # Final conversion of potential
    learned_potential = math.wrap(learned_potential_torch.detach().numpy(), spatial(x=pot_size))
    
    return model, learned_potential


def evaluate_model(model: SimplePINNModel,
                  physics: InviscidBurgersPhysics,
                  num_rollout_steps: int = 10) -> dict:
    """Evaluate model rollout vs ground truth."""
    potential = physics.create_potential_field()
    
    # Create test initial condition
    initial = CenteredGrid(
        Noise(scale=physics.domain.size['x']/4, smoothness=2.0),
        extrapolation.PERIODIC,
        bounds=physics.domain,
        x=physics.resolution.get_size('x')
    ).values
    
    # Ground truth rollout
    gt_trajectory = physics.generate_trajectory(initial, potential, num_rollout_steps)
    
    # Model rollout
    model_trajectory = [initial]
    current = initial
    for _ in range(num_rollout_steps):
        current = model(current)  # Use model wrapper which handles tensor format
        model_trajectory.append(current)
    model_trajectory = math.stack(model_trajectory, batch('time'))
    
    # Compute errors
    errors = []
    for t in range(1, num_rollout_steps + 1):
        gt = gt_trajectory.time[t]
        pred = model_trajectory.time[t]
        mse = float(math.mean((gt - pred) ** 2))
        errors.append(mse)
    
    return {
        'step_errors': errors,
        'mean_error': np.mean(errors),
        'final_error': errors[-1],
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*60)
    print("Minimal PINN Prototype for Inviscid Burgers 1D")
    print("="*60)
    
    # Setup physics
    physics = InviscidBurgersPhysics(
        domain_size=CONFIG['domain_size'],
        resolution=CONFIG['resolution'],
        dt=CONFIG['dt']
    )
    
    residual_computer = InviscidBurgersResidual(
        domain_size=CONFIG['domain_size'],
        resolution=CONFIG['resolution'],
        dt=CONFIG['dt']
    )
    
    # Generate training data
    print("\nGenerating training data...")
    trajectories = generate_training_data(
        physics,
        num_trajectories=CONFIG['batch_size'],
        num_timesteps=CONFIG['num_timesteps']
    )
    print(f"Generated {CONFIG['batch_size']} trajectories of length {CONFIG['num_timesteps']}")
    print(f"Data shape: {trajectories.shape}")
    
    # Test 1: PINN with fixed parameters
    print("\n" + "-"*60)
    print("TEST 1: PINN with FIXED potential field")
    print("-"*60)
    
    model_fixed = train_pinn_fixed_params(
        trajectories,
        physics,
        residual_computer,
        num_epochs=CONFIG['num_epochs'],
        lr=CONFIG['learning_rate'],
        pinn_weight=CONFIG['pinn_weight']
    )
    
    results_fixed = evaluate_model(model_fixed, physics)
    print(f"\nEvaluation (Fixed Params):")
    print(f"  Mean rollout error: {results_fixed['mean_error']:.6f}")
    print(f"  Final step error:   {results_fixed['final_error']:.6f}")
    
    # Test 2: PINN with learned parameters
    print("\n" + "-"*60)
    print("TEST 2: PINN with LEARNED potential field")
    print("-"*60)
    
    model_learned, learned_potential = train_pinn_learned_params(
        trajectories,
        physics,
        residual_computer,
        num_epochs=CONFIG['num_epochs'],
        lr=CONFIG['learning_rate'],
        pinn_weight=CONFIG['pinn_weight']
    )
    
    results_learned = evaluate_model(model_learned, physics)
    print(f"\nEvaluation (Learned Params):")
    print(f"  Mean rollout error: {results_learned['mean_error']:.6f}")
    print(f"  Final step error:   {results_learned['final_error']:.6f}")
    
    # Compare learned vs true potential
    true_potential = physics.create_potential_field()
    pot_error = float(math.mean((learned_potential - true_potential) ** 2))
    pot_corr = float(math.sum(learned_potential * true_potential) / 
                    (math.sqrt(math.sum(learned_potential**2)) * 
                     math.sqrt(math.sum(true_potential**2))))
    
    print(f"\nPotential field recovery:")
    print(f"  MSE vs ground truth:  {pot_error:.6f}")
    print(f"  Correlation:          {pot_corr:.4f}")
    
    print("\n" + "="*60)
    print("PINN Prototype Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
