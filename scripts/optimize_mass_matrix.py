"""Script to optimize the mass/mobility matrix for Langevin dynamics.

This script demonstrates how to optimize the preconditioning matrix R (where D = RR^T)
to maximize the probability of crossing a barrier in a 2D Double Well potential.

The optimization objective is to minimize the squared distance to the target state
at the end of a fixed-time simulation.

This uses the `PreconditionedOverdampedLangevin` integrator and backpropagation through time (BPTT).
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from uni_diffsim.potentials import DoubleWell2D
from uni_diffsim.integrators import PreconditionedOverdampedLangevin
from uni_diffsim.plotting import apply_style, COLORS

def optimize_mass_matrix(
    n_steps: int = 1000,
    dt: float = 0.01,
    n_epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    seed: int = 42
):
    torch.manual_seed(seed)
    apply_style()

    # Setup potential: High barrier in X, Strong confinement in Y
    # Barrier height 2.5 with kT=1.0 makes crossing rare with low diffusion
    potential = DoubleWell2D(barrier_height=2.5, k_y=5.0)

    # Target state: Right minimum
    target_state = torch.tensor([1.0, 0.0])
    start_state = torch.tensor([-1.0, 0.0]) # Left minimum

    # Setup integrator with learnable R
    dim = 2
    # Initialize R with LOW diffusion (0.5)
    # This corresponds to D=0.25, Mass=4.0
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
        x0 = start_state.unsqueeze(0).expand(batch_size, -1) + 0.1 * torch.randn(batch_size, dim)
        traj_init = integrator.run(x0, potential.force, dt, n_steps)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Start batch of particles at Left Minima
        x0 = start_state.unsqueeze(0).expand(batch_size, -1) + 0.1 * torch.randn(batch_size, dim)

        # Run simulation
        traj = integrator.run(x0, potential.force, dt, n_steps)
        final_x = traj[-1]

        # Objective: Minimize squared distance to target (Right Minima)
        dist_sq = ((final_x - target_state)**2).sum(dim=-1)
        loss = dist_sq.mean()

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(integrator.parameters(), 1.0)

        optimizer.step()

        losses.append(loss.item())

        if epoch % 5 == 0:
            # Check crossing fraction
            crossed = (final_x[:, 0] > 0).float().mean()
            print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}, Crossing Rate = {crossed.item():.2%}")

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
        x0 = start_state.unsqueeze(0).expand(batch_size, -1) + 0.1 * torch.randn(batch_size, dim)
        traj_final = integrator.run(x0, potential.force, dt, n_steps)

    # Visualization
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Potential Surface
        ax = axes[0]
        grid_x = torch.linspace(-2.0, 2.0, 100)
        grid_y = torch.linspace(-1.5, 1.5, 100)
        X, Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        XY = torch.stack([X, Y], dim=-1)
        Z = potential.energy(XY).detach()
        Z = Z.clamp(max=10)

        ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
        ax.contour(X, Y, Z, levels=20, colors='k', alpha=0.1)

        # Plot Initial Trajectories
        count = 0
        for i in range(batch_size):
            if count < 20:
                ax.plot(traj_init[:, i, 0], traj_init[:, i, 1], color=COLORS['gray'], alpha=0.5, lw=1)
                count += 1

        ax.scatter(start_state[0], start_state[1], marker='*', color=COLORS['blue'], s=100, label='Start')
        ax.scatter(target_state[0], target_state[1], marker='x', color=COLORS['red'], s=100, label='Target')
        ax.set_title("Initial Trajectories (High Mass)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.5)

        # Plot Final Trajectories
        ax = axes[1]
        ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
        ax.contour(X, Y, Z, levels=20, colors='k', alpha=0.1)

        count = 0
        for i in range(batch_size):
             if count < 20:
                ax.plot(traj_final[:, i, 0], traj_final[:, i, 1], color=COLORS['optimal'], alpha=0.6, lw=1)
                count += 1

        ax.scatter(start_state[0], start_state[1], marker='*', color=COLORS['blue'], s=100)
        ax.scatter(target_state[0], target_state[1], marker='x', color=COLORS['red'], s=100)
        ax.set_title("Optimized Trajectories (Anisotropic)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.5)

        # Plot Loss
        ax = axes[2]
        if losses:
            ax.plot(losses, color=COLORS['blue'])
        ax.set_title("Optimization Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Squared Distance")

        plt.tight_layout()
        plt.savefig("mass_matrix_optimization.png")
        print("Plot saved to mass_matrix_optimization.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    optimize_mass_matrix()
