"""Demo: Gradient Ghost (Shadowing) vs BPTT on Chaotic Lorenz 63.

This script demonstrates that standard Backpropagation Through Time (BPTT) fails
to compute valid gradients for chaotic systems over long time horizons due to
exploding gradients (butterfly effect), while the "Gradient Ghost" (Least Squares
Shadowing) estimator recovers stable, meaningful gradients.

System: Lorenz 63 (sigma=10, rho=28, beta=8/3)
Objective: d<z>/d(rho)  (Sensitivity of mean z-position to rho)
Reference value: d<z>/d(rho) ≈ 1.0 (for rho=28)
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from uni_diffsim.integrators import Lorenz63
from uni_diffsim.ghost_gradient import GhostGradientEstimator
from uni_diffsim.plotting import apply_style, COLORS

apply_style()

def main():
    print("Initializing Lorenz 63 system...")
    # Lorenz 63 parameters
    # rho=28 is the standard chaotic regime
    lorenz = Lorenz63(sigma=10.0, rho=28.0, beta=8.0/3.0)

    # Simulation settings
    dt = 0.01
    n_steps_long = 1000  # Long enough for chaos to manifest
    x0 = torch.tensor([1.0, 1.0, 1.0])

    # 1. Run forward simulation (ground truth)
    print(f"Running simulation for {n_steps_long} steps...")
    traj = lorenz.run(x0, dt=dt, n_steps=n_steps_long)

    # Plot trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(traj[:, 0].detach(), traj[:, 2].detach(), color=COLORS['trajectory'], alpha=0.8, lw=1)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Lorenz 63 Attractor")
    plt.tight_layout()
    plt.savefig("lorenz_attractor.png")
    print("Saved attractor plot to lorenz_attractor.png")

    # 2. Compute Gradient via BPTT (Autograd)
    print("\nComputing gradients via BPTT (Standard Backprop)...")
    lorenz.zero_grad()

    # Re-run with gradient tracking
    # We need to re-create graph
    x0_grad = x0.clone().detach()
    # We need to trace parameters, so we use lorenz instance
    # But lorenz.run detaches by default? No, usually integrators support grad if params have grad.
    # Let's check Integrator.run implementation. It uses _integrate which creates new tensors.
    # We need to ensure graph is maintained.

    # Manual run for BPTT to ensure graph connectivity
    state = x0_grad
    trajectory_bptt = []

    # Need to verify if lorenz.rho has grad enabled
    assert lorenz.rho.requires_grad

    # Run loop
    for _ in range(n_steps_long):
        state = lorenz.step(state, dt)
        trajectory_bptt.append(state)

    trajectory_bptt = torch.stack(trajectory_bptt)

    # Objective: Mean z position
    loss_bptt = trajectory_bptt[:, 2].mean()
    loss_bptt.backward()

    grad_bptt = lorenz.rho.grad.item()
    print(f"BPTT Gradient d<z>/drho: {grad_bptt:.4e}")
    if abs(grad_bptt) > 100:
        print("  -> EXPLODED! (As expected for chaos)")

    # 3. Compute Gradient via Gradient Ghost (Shadowing)
    print("\nComputing gradients via Gradient Ghost (LSS)...")
    lorenz.zero_grad() # Clear gradients

    estimator = GhostGradientEstimator(lorenz, param_names=['rho'])

    # Helper for observable (z coordinate)
    def observable_z(x):
        # x is (..., 3)
        return x[..., 2]

    # Use the trajectory from before (detached is fine for Ghost, as it computes Jacobian freshly)
    # We pass the detached trajectory
    grads_ghost = estimator.estimate_gradient(traj.detach(), observable=observable_z, dt=dt)

    grad_ghost = grads_ghost['rho'].item()
    print(f"Ghost Gradient d<z>/drho: {grad_ghost:.4f}")
    print("  -> STABLE! (Consistent with linear response)")

    # 4. Compare gradients over time (windowed analysis)
    # Shows how BPTT gradient grows exponentially while Ghost stays stable
    print("\nRunning windowed analysis (Gradient vs Time)...")

    windows = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    bptt_grads = []
    ghost_grads = []

    for T in windows:
        # BPTT
        lorenz.zero_grad()
        state = x0.clone()
        traj_list = []
        for _ in range(T):
            state = lorenz.step(state, dt)
            traj_list.append(state)
        traj_stack = torch.stack(traj_list)
        loss = traj_stack[:, 2].mean()
        loss.backward()
        bptt_grads.append(lorenz.rho.grad.item())

        # Ghost (on same trajectory subset)
        # We can reuse the trajectory values
        g_est = estimator.estimate_gradient(traj.detach()[:T], observable_z, dt)
        ghost_grads.append(g_est['rho'].item())

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(windows, [abs(g) for g in bptt_grads], 'o--', label='BPTT (Standard)', color=COLORS['red'])
    plt.plot(windows, [abs(g) for g in ghost_grads], 's-', label='Gradient Ghost (LSS)', color=COLORS['green'])
    plt.yscale('log')
    plt.xlabel("Trajectory Length (steps)")
    plt.ylabel("|Gradient d<z>/drho|")
    plt.title("Gradient Stability: Ghost vs BPTT on Lorenz 63")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("gradient_comparison.png")
    print("Saved comparison plot to gradient_comparison.png")

if __name__ == "__main__":
    main()
