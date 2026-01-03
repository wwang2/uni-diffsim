
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from uni_diffsim.potentials import DimerWCA
from uni_diffsim.integrators import OverdampedLangevin
from uni_diffsim.gradient_estimators import reinforce_gradient
from uni_diffsim.plotting import apply_style, COLORS

# Apply shared plotting style
apply_style()

def init_lattice(n_solvent: int, box_size: float) -> torch.Tensor:
    """Initialize solvent particles on a simple cubic lattice."""
    k = int(np.ceil(n_solvent**(1/3)))
    if k**3 < n_solvent: k += 1

    spacing = box_size / k
    # Centered grid
    x = torch.linspace(-box_size/2 + spacing/2, box_size/2 - spacing/2, k)
    grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
    positions = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    # Take first n_solvent particles
    if positions.shape[0] > n_solvent:
        positions = positions[:n_solvent]

    # Add VERY small noise
    positions += torch.randn_like(positions) * (spacing * 0.01)

    return positions

def create_system(n_batch: int = 16, device: str = "cpu") -> Tuple[DimerWCA, torch.Tensor]:
    """Create DimerWCA system and initial state."""
    # DimerWCA parameters
    n_solvent = 64
    density = 0.5
    h = 1.0

    dimer = DimerWCA(n_solvent=n_solvent, density=density, h=h, eps=1.0)
    dimer.to(device)

    box_size = dimer.box_size[0].item()
    print(f"System: {n_solvent} solvent + 2 dimer. Box size: {box_size:.2f}")

    solvent_pos = init_lattice(n_solvent, box_size).to(device)

    r0 = dimer.r0.item()

    # Dimer particles
    p0 = torch.tensor([-r0/2, 0.0, 0.0], device=device)
    p1 = torch.tensor([r0/2, 0.0, 0.0], device=device)

    # Remove solvent particles overlapping with dimer
    # Just shift them to the end of the box if they overlap
    dist0 = (solvent_pos - p0).norm(dim=1)
    dist1 = (solvent_pos - p1).norm(dim=1)
    min_dist = 1.2 # slightly larger than sigma

    mask_keep = (dist0 >= min_dist) & (dist1 >= min_dist)
    n_removed = (~mask_keep).sum().item()

    if n_removed > 0:
        print(f"Removing {n_removed} overlapping solvent particles.")
        solvent_pos = solvent_pos[mask_keep]

    x0_single = torch.cat([p0.unsqueeze(0), p1.unsqueeze(0), solvent_pos], dim=0) # (2+n_solvent_actual, 3)
    x0 = x0_single.unsqueeze(0).repeat(n_batch, 1, 1)

    return dimer, x0

def relax_system(x0: torch.Tensor, potential: DimerWCA, n_steps: int = 500, dt: float = 0.001):
    """Energy minimization / relaxation."""
    print(f"Relaxing system for {n_steps} steps (dt={dt})...")
    # Use very small dt for first few steps to resolve bad contacts
    # Increased gamma further to quench high velocities
    integrator = OverdampedLangevin(gamma=50.0, kT=0.1)

    with torch.no_grad():
        # Soft start with capped forces? DimerWCA doesn't support force capping easily.
        # Use extremely small dt initially.
        x = x0
        current_dt = dt * 0.01
        for i in range(20):
            x = integrator.run(x, potential.force, dt=current_dt, n_steps=50, final_only=True)[0]
            if torch.isnan(x).any():
                print("NaN during soft start!")
                return x0 # Return bad state to fail gracefully later
            current_dt *= 1.2 # Slowly increase dt
            if current_dt > dt: current_dt = dt

        # Normal relaxation
        x_relaxed = integrator.run(x, potential.force, dt=dt, n_steps=n_steps, final_only=True)[0]

        return x_relaxed

def run_solvation_design_demo():
    print("================================================================")
    print("  Differentiable Simulation Demo: WCA Solvation Design")
    print("================================================================")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # --- Setup ---
    n_batch = 8
    dimer, x0 = create_system(n_batch, device)

    # Check initial energy
    u_init = dimer.energy(x0)
    print(f"Initial Max Energy: {u_init.max().item():.2e}")
    if u_init.max() > 1e4:
        print("Warning: High initial energy, relaxation is critical.")

    # Relax initial state
    x = relax_system(x0, dimer, n_steps=500, dt=0.001)
    u_relaxed = dimer.energy(x)
    print(f"Relaxed Max Energy: {u_relaxed.max().item():.2f}")

    if torch.isnan(x).any():
        print("Error: NaN detected after relaxation!")
        return

    # Integrator for simulation
    kT = 1.0
    integrator = OverdampedLangevin(gamma=5.0, kT=kT).to(device) # Higher gamma for stability

    # Optimizer for eps
    if not dimer.eps.requires_grad:
        dimer.eps.requires_grad_(True)
    optimizer = optim.Adam([dimer.eps], lr=0.1)

    # Simulation settings
    n_steps_per_epoch = 1000
    burn_in = 200
    dt = 0.001 # Reduced dt for WCA stability
    n_epochs = 10

    # Constants
    r0 = dimer.r0.item()
    w = dimer.w.item()
    barrier_dist = r0 + w
    print(f"Compact < {barrier_dist:.2f} < Extended")

    history_eps = []
    history_prob = []

    print("\nStarting Optimization Loop...")
    print(f"{'Epoch':<6} | {'Epsilon':<8} | {'P(Extended)':<12} | {'Grad(eps)':<10}")
    print("-" * 50)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        x = x.detach()

        with torch.no_grad():
            traj = integrator.run(x, dimer.force, dt=dt, n_steps=n_steps_per_epoch, store_every=10)

            if torch.isnan(traj[-1]).any():
                print(f"Epoch {epoch}: Simulation exploded (NaN). Stopping.")
                break

            x = traj[-1]
            samples = traj[burn_in//10:]

        def get_bond_length(pos):
            return (pos[..., 0, :] - pos[..., 1, :]).norm(dim=-1)

        def observable_fn(pos):
            d = get_bond_length(pos)
            return (d > barrier_dist).float()

        # Estimate Gradient via REINFORCE
        grads = reinforce_gradient(samples, dimer, observable=observable_fn, beta=1.0/kT)

        if 'eps' in grads and grads['eps'] is not None:
            # Clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_([dimer.eps], 1.0)

            dimer.eps.grad = -grads['eps'] # Ascent

            # Clip explicit gradient value if needed
            if dimer.eps.grad.abs() > 1.0:
                 dimer.eps.grad = dimer.eps.grad.clamp(-1.0, 1.0)

            optimizer.step()

            with torch.no_grad():
                dimer.eps.clamp_(min=0.1, max=5.0) # Prevent eps from exploding or vanishing

            grad_val = grads['eps'].item()
        else:
            grad_val = 0.0

        with torch.no_grad():
            bond_lengths = get_bond_length(samples)
            p_extended = (bond_lengths > barrier_dist).float().mean().item()

        history_eps.append(dimer.eps.item())
        history_prob.append(p_extended)

        print(f"{epoch+1:<6d} | {dimer.eps.item():<8.3f} | {p_extended:<12.3f} | {grad_val:<10.3e}")

    print("\nOptimization Complete.")
    print(f"Final Epsilon: {dimer.eps.item():.3f}")
    if len(history_prob) > 0:
        print(f"Final P(Extended): {history_prob[-1]:.3f}")

    # Plotting with consistent style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Parameter Optimization
    ax1.plot(history_eps, 'o-', color=COLORS['blue'], label=r'Solvent $\epsilon$', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'Solvent $\epsilon$')
    ax1.set_title('Parameter Optimization')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: Target Objective
    ax2.plot(history_prob, 's-', color=COLORS['orange'], label='P(Extended)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Probability')
    ax2.set_title('Target Objective')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('wca_optimization.png', dpi=150)
    print("Plot saved to wca_optimization.png")

if __name__ == "__main__":
    run_solvation_design_demo()
