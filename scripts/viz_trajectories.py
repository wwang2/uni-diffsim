"""Visualization of System Trajectories.

This script generates a visual comparison of the dynamics between:
1. Overdamped Langevin (Stochastic, Diffusive)
2. Nose-Hoover (Deterministic, Chaotic)

on a Double Well potential. It helps visualize the "what is going on"
behind the variance-bias trade-off exploration.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from uni_diffsim.potentials import DoubleWell
from uni_diffsim.integrators import OverdampedLangevin, NoseHoover
from uni_diffsim.plotting import apply_style, COLORS, get_assets_dir

# Configuration
BARRIER_HEIGHT = 2.0
BETA = 1.0
DT = 0.05
N_STEPS = 2000

def get_theoretical_density(potential, beta, x_range=(-2.5, 2.5), n_points=1000):
    x = torch.linspace(x_range[0], x_range[1], n_points)
    with torch.no_grad():
        U = potential.energy(x.unsqueeze(-1))
        prob = torch.exp(-beta * U)
        prob = prob / (prob.sum() * (x[1] - x[0]))
    return x.numpy(), prob.numpy()

def main():
    apply_style()

    # Setup
    potential = DoubleWell(barrier_height=BARRIER_HEIGHT)

    # --- 1. Langevin Dynamics ---
    langevin = OverdampedLangevin(gamma=1.0, kT=1.0/BETA)
    x0 = torch.zeros(1, 1) # Start at barrier

    torch.manual_seed(42) # Fixed seed for reproducibility
    traj_langevin = langevin.run(x0, potential.force, dt=DT, n_steps=N_STEPS)
    traj_langevin = traj_langevin.detach().squeeze().numpy()

    # --- 2. Nose-Hoover Dynamics ---
    nh = NoseHoover(kT=1.0/BETA, mass=1.0, Q=1.0)
    x0 = torch.zeros(1, 1)
    v0 = torch.randn(1, 1) * np.sqrt(1.0/BETA)

    torch.manual_seed(42)
    traj_nh_x, traj_nh_v = nh.run(x0, v0, potential.force, dt=DT, n_steps=N_STEPS)
    traj_nh_x = traj_nh_x.detach().squeeze().numpy()
    traj_nh_v = traj_nh_v.detach().squeeze().numpy()

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)

    # Row 1: Langevin
    ax_lang_ts = fig.add_subplot(gs[0, 0])
    ax_lang_hist = fig.add_subplot(gs[0, 1])

    # Time Series
    t = np.arange(len(traj_langevin)) * DT
    ax_lang_ts.plot(t, traj_langevin, color=COLORS['overdamped'], alpha=0.8, lw=1)
    ax_lang_ts.set_title("Langevin: Time Series (Stochastic Diffusion)")
    ax_lang_ts.set_xlabel("Time")
    ax_lang_ts.set_ylabel("Position x")
    ax_lang_ts.set_ylim(-2.5, 2.5)
    # Mark wells
    ax_lang_ts.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax_lang_ts.axhline(-1.0, color='gray', ls='--', alpha=0.3)

    # Histogram vs Theory
    x_theory, p_theory = get_theoretical_density(potential, BETA)
    ax_lang_hist.hist(traj_langevin, bins=30, density=True, orientation='horizontal',
                      color=COLORS['overdamped'], alpha=0.4, label='Sampled')
    ax_lang_hist.plot(p_theory, x_theory, color=COLORS['theory'], lw=2, label='Theoretical')
    ax_lang_hist.set_title("Langevin: Equilibrium Distribution")
    ax_lang_hist.set_xlabel("Density")
    ax_lang_hist.set_ylim(-2.5, 2.5)
    ax_lang_hist.set_yticklabels([]) # Share y-axis visually
    ax_lang_hist.legend()

    # Row 2: Nose-Hoover
    ax_nh_ts = fig.add_subplot(gs[1, 0])
    ax_nh_phase = fig.add_subplot(gs[1, 1])

    # Time Series
    ax_nh_ts.plot(t, traj_nh_x, color=COLORS['nh'], alpha=0.8, lw=1)
    ax_nh_ts.set_title("Nose-Hoover: Time Series (Chaotic Oscillation)")
    ax_nh_ts.set_xlabel("Time")
    ax_nh_ts.set_ylabel("Position x")
    ax_nh_ts.set_ylim(-2.5, 2.5)
    ax_nh_ts.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax_nh_ts.axhline(-1.0, color='gray', ls='--', alpha=0.3)

    # Phase Space
    ax_nh_phase.plot(traj_nh_x, traj_nh_v, color=COLORS['nh'], alpha=0.5, lw=0.5)
    ax_nh_phase.set_title("Nose-Hoover: Phase Space (Ergodic Orbit)")
    ax_nh_phase.set_xlabel("Position x")
    ax_nh_phase.set_ylabel("Velocity v")
    ax_nh_phase.set_xlim(-2.5, 2.5)
    ax_nh_phase.set_ylim(-3, 3)

    # Add potential contours to phase space background?
    # H(x,v) = v^2/2 + U(x). Contours of H.
    X, V = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-3, 3, 100))
    U_grid = BARRIER_HEIGHT * (X**2 - 1)**2
    H_grid = 0.5 * V**2 + U_grid
    ax_nh_phase.contour(X, V, H_grid, levels=10, colors='gray', alpha=0.2, linewidths=0.5)

    plt.tight_layout()

    assets_dir = get_assets_dir()
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, "trajectory_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()
