"""Script to optimize the mass/mobility matrix for Langevin dynamics.

This script demonstrates how to optimize the preconditioning matrix R (where D = RR^T)
to maximize the probability of crossing a barrier in the Müller-Brown potential.

The optimization objective is to minimize the squared distance to the target state B
at the end of a fixed-time simulation.

This uses the `PreconditionedOverdampedLangevin` integrator and backpropagation through time (BPTT).
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from uni_diffsim.potentials import MullerBrown
from uni_diffsim.integrators import PreconditionedOverdampedLangevin
from uni_diffsim.plotting import apply_style, COLORS

def optimize_mass_matrix(
    n_steps: int = 200,
    dt: float = 1e-4, # Very small timestep for stability
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    seed: int = 42
):
    torch.manual_seed(seed)
    apply_style()

    # Setup potential
    potential = MullerBrown()

    # Target state (approximate minimum B of Muller-Brown)
    target_state = torch.tensor([0.623, 0.028])
    start_state = torch.tensor([-0.558, 1.441]) # Minima A

    # Setup integrator with learnable R
    # Initialize R close to identity
    dim = 2
    # Initialize with slightly smaller scale to be safe
    R_init = torch.eye(dim) * 0.5
    integrator = PreconditionedOverdampedLangevin(dim=dim, kT=1.0, R=R_init)

    # Optimizer
    optimizer = torch.optim.Adam(integrator.parameters(), lr=learning_rate)

    # Tracking
    losses = []

    print(f"Starting optimization for {n_epochs} epochs...")
    print(f"Initial diffusion matrix:\n{integrator.diffusion_matrix.detach().numpy()}")

    # Store initial trajectory for visualization
    with torch.no_grad():
        x0 = start_state.unsqueeze(0).expand(batch_size, -1) + 0.05 * torch.randn(batch_size, dim)
        traj_init = integrator.run(x0, potential.force, dt, n_steps)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Start batch of particles at Minima A
        x0 = start_state.unsqueeze(0).expand(batch_size, -1) + 0.05 * torch.randn(batch_size, dim)

        # Run simulation
        traj = integrator.run(x0, potential.force, dt, n_steps)
        final_x = traj[-1]

        # Check for NaNs
        if torch.isnan(final_x).any():
            print(f"Epoch {epoch:03d}: NaNs detected in trajectory! Skipping update.")
            # If NaNs, we might need to reset parameters or reduce LR, but here we just skip
            continue

        # Objective: Minimize squared distance to target
        dist_sq = ((final_x - target_state)**2).sum(dim=-1)
        loss = dist_sq.mean()

        # Regularization: Penalize large diffusion to prevent instability
        D = integrator.diffusion_matrix
        loss += 0.1 * D.norm()

        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(integrator.parameters(), 0.5)

        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}")

    if len(losses) > 0:
        print(f"Final Loss = {losses[-1]:.4f}")

    D_final = integrator.diffusion_matrix.detach()
    print(f"Final diffusion matrix D = R R^T:\n{D_final.numpy()}")

    try:
        M_final = integrator.mass_matrix.detach()
        print(f"Final mass matrix M = (R R^T)^-1:\n{M_final.numpy()}")
    except Exception:
        pass

    # Store final trajectory
    with torch.no_grad():
        x0 = start_state.unsqueeze(0).expand(batch_size, -1) + 0.05 * torch.randn(batch_size, dim)
        traj_final = integrator.run(x0, potential.force, dt, n_steps)

    # Visualization
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Potential Surface
        ax = axes[0]
        grid_x = torch.linspace(-1.8, 1.2, 100)
        grid_y = torch.linspace(-0.5, 2.2, 100)
        X, Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        XY = torch.stack([X, Y], dim=-1)
        Z = potential.energy(XY).detach()
        Z = Z.clamp(max=200)

        ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.3)
        ax.contour(X, Y, Z, levels=30, colors='k', alpha=0.1)

        # Plot Initial Trajectories
        valid_idx = ~torch.isnan(traj_init[-1, :, 0])
        count = 0
        for i in range(batch_size):
            if valid_idx[i] and count < 10:
                ax.plot(traj_init[:, i, 0], traj_init[:, i, 1], color=COLORS['gray'], alpha=0.5, lw=1)
                count += 1

        ax.scatter(start_state[0], start_state[1], marker='*', color=COLORS['blue'], s=100, label='Start')
        ax.scatter(target_state[0], target_state[1], marker='x', color=COLORS['red'], s=100, label='Target')
        ax.set_title("Initial Trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.8, 1.2)
        ax.set_ylim(-0.5, 2.2)

        # Plot Final Trajectories
        ax = axes[1]
        ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.3)
        ax.contour(X, Y, Z, levels=30, colors='k', alpha=0.1)

        valid_idx_final = ~torch.isnan(traj_final[-1, :, 0])
        count = 0
        for i in range(batch_size):
             if valid_idx_final[i] and count < 10:
                ax.plot(traj_final[:, i, 0], traj_final[:, i, 1], color=COLORS['optimal'], alpha=0.6, lw=1)
                count += 1

        ax.scatter(start_state[0], start_state[1], marker='*', color=COLORS['blue'], s=100)
        ax.scatter(target_state[0], target_state[1], marker='x', color=COLORS['red'], s=100)
        ax.set_title("Optimized Trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.8, 1.2)
        ax.set_ylim(-0.5, 2.2)

        # Plot Loss
        ax = axes[2]
        if losses:
            ax.plot(losses, color=COLORS['blue'])
        ax.set_title("Optimization Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        plt.tight_layout()
        plt.savefig("mass_matrix_optimization.png")
        print("Plot saved to mass_matrix_optimization.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    optimize_mass_matrix()
