"""Demo: Gradient Ghost (Shadowing) on Henon-Heiles Chaotic Potential.

This script demonstrates the "Gradient Ghost" (Least Squares Shadowing) estimator
on a classic Hamiltonian chaos system: the Henon-Heiles potential.

U(x, y) = 0.5(x^2 + y^2) + lambda * (x^2*y - y^3/3)

At high energies (E > 1/6), the motion becomes chaotic.
We compare BPTT and Ghost Gradients for sensitivity to lambda.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from uni_diffsim.potentials import HenonHeiles
from uni_diffsim.integrators import VelocityVerlet
from uni_diffsim.ghost_gradient import GhostGradientEstimator
from uni_diffsim.plotting import apply_style, COLORS

apply_style()

class MDDynamicalSystem(nn.Module):
    """Wraps an Integrator and Potential into a single dynamical system z_{t+1} = f(z_t).

    State z is [x, v].
    """
    def __init__(self, integrator, potential):
        super().__init__()
        self.integrator = integrator
        self.potential = potential

    def step(self, z, dt):
        """Step function for combined state z = [x, v]."""
        # z: (..., 2*dim)
        # Assuming event_dim is usually 1 or 2.
        # For HenonHeiles (dim=2), z is 4D.
        dim = z.shape[-1] // 2
        x, v = z[..., :dim], z[..., dim:]

        # Integrator step
        x_new, v_new = self.integrator.step(x, v, self.potential.force, dt)

        return torch.cat([x_new, v_new], dim=-1)

def main():
    print("Initializing Henon-Heiles system...")
    # Lambda=1.0. Energy E=1/6 is critical energy for escape.
    # We want bound chaos.
    # Actually, for lambda=1, escape energy is 1/6 ~ 0.1667.
    # Chaos sets in before escape.
    hh = HenonHeiles(lam=1.0)

    # Velocity Verlet integrator
    integrator = VelocityVerlet(mass=1.0)

    # Combined system
    system = MDDynamicalSystem(integrator, hh)

    # Simulation settings
    dt = 0.1
    n_steps = 1000

    # Initial condition (Energy ~ 0.15, close to critical 0.166)
    x0 = torch.tensor([0.0, 0.0])
    v0 = torch.tensor([0.45, 0.25]) # Kinetic energy ~ 0.13

    z0 = torch.cat([x0, v0])

    # 1. Run forward simulation
    print(f"Running simulation for {n_steps} steps...")
    traj_z = []
    z = z0.clone()
    for _ in range(n_steps):
        z = system.step(z, dt)
        traj_z.append(z)
    traj_z = torch.stack(traj_z)

    # Plot trajectory (x vs y)
    plt.figure(figsize=(8, 8))
    plt.plot(traj_z[:, 0].detach(), traj_z[:, 1].detach(), color=COLORS['trajectory'], alpha=0.8, lw=1)
    # Plot potential contours
    grid_x = torch.linspace(-1, 1, 100)
    grid_y = torch.linspace(-1, 1, 100)
    X, Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    XY = torch.stack([X, Y], dim=-1)
    Z = hh.energy(XY).detach()
    plt.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Henon-Heiles Trajectory")
    plt.tight_layout()
    plt.savefig("henon_heiles_traj.png")
    print("Saved trajectory plot to henon_heiles_traj.png")

    # 2. Compute Gradient via BPTT
    print("\nComputing gradients via BPTT...")
    hh.zero_grad()

    # Re-run for BPTT
    # Note: Need to ensure parameters are tracked.
    # BPTT needs us to execute operations on parameters.
    # The system.step calls potential.force which uses hh.lam.

    # We trace manually for BPTT
    z = z0.clone().detach() # No grad w.r.t. initial state
    traj_list = []

    # Enable grad on lam? It's already nn.Parameter.
    # We just need to make sure the loop constructs the graph.

    for _ in range(n_steps):
        # We must call system.step which uses self.potential which has self.lam
        z = system.step(z, dt)
        traj_list.append(z)

    traj_stack = torch.stack(traj_list)

    # Objective: Mean x^2 (just an arbitrary observable)
    loss_bptt = (traj_stack[..., 0]**2).mean()
    loss_bptt.backward()

    grad_bptt = hh.lam.grad.item()
    print(f"BPTT Gradient d<x^2>/dlam: {grad_bptt:.4e}")

    # 3. Compute Gradient via Gradient Ghost
    print("\nComputing gradients via Gradient Ghost...")
    hh.zero_grad()

    # Estimator needs param names.
    # Note: system wraps integrator and potential.
    # GhostGradientEstimator iterates named_parameters of "integrator".
    # Here, we pass "system" as the "integrator" to GhostGradientEstimator.
    # System has parameters from both.

    estimator = GhostGradientEstimator(system, param_names=['potential.lam'])

    # Observable: x^2
    def obs_x2(z):
        # z: (..., 2*dim)
        # Handle batch dimension if present
        if z.ndim > 1:
            return z[..., 0]**2
        return z[0]**2

    # Run estimator
    grads_ghost = estimator.estimate_gradient(traj_z.detach(), observable=obs_x2, dt=dt)

    # Note: keys in grads_ghost match param_names
    grad_ghost = grads_ghost['potential.lam'].item()
    print(f"Ghost Gradient d<x^2>/dlam: {grad_ghost:.4f}")

    # 4. Compare
    print("\nComparison:")
    print(f"  BPTT:  {grad_bptt:.4e}")
    print(f"  Ghost: {grad_ghost:.4f}")

if __name__ == "__main__":
    main()
