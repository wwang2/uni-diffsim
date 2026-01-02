"""Demo: Variance-Bias-Chaos Trade-off in Gradient Estimation.

This script explores the trade-offs between different gradient estimation methods
(BPTT, REINFORCE, Girsanov, Adjoint) across different regimes (Short/Long T, Chaotic/Non-Chaotic).

It systematically compares:
- Bias (w.r.t. true equilibrium gradient)
- Variance (across seeds)
- MSE (Bias^2 + Variance)
- Computational Cost (Wall-clock time)

The target problem is optimizing the barrier height of a Double Well potential
to minimize the potential energy (or other observable). Here we compute gradients
of <x^2> w.r.t barrier height.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import os
from collections import defaultdict

from uni_diffsim.potentials import DoubleWell
from uni_diffsim.integrators import OverdampedLangevin, NoseHoover
from uni_diffsim.gradient_estimators import ReinforceEstimator, GirsanovEstimator, CheckpointedNoseHoover
from uni_diffsim.plotting import apply_style, COLORS, get_assets_dir

# Configuration
BETA = 1.0
BARRIER_HEIGHT = 2.0
DT = 0.05
N_TRIALS = 5
REGIMES = {
    "Short T": 500,
    "Long T": 2000
}

# -----------------------------------------------------------------------------
# Ground Truth Calculation
# -----------------------------------------------------------------------------

class GroundTruth:
    """Computes exact gradients via 1D numerical integration."""

    def __init__(self, barrier_height, beta=1.0, grid_size=2000, bound=5.0):
        self.a = barrier_height
        self.beta = beta
        self.x = torch.linspace(-bound, bound, grid_size)
        self.dx = self.x[1] - self.x[0]

    def compute_gradient(self):
        """Compute d<x^2>/da exactly."""
        x = self.x
        a = self.a

        # Potential: U = a(x^2 - 1)^2
        U = a * (x**2 - 1)**2

        # Probability density
        prob_unnorm = torch.exp(-self.beta * U)
        Z = prob_unnorm.sum() * self.dx
        prob = prob_unnorm / Z

        # Observable O = x^2
        O = x**2
        E_O = (O * prob).sum() * self.dx

        # Gradient of potential w.r.t parameter a: dU/da = (x^2 - 1)^2
        dU_da = (x**2 - 1)**2

        # Covariance term: Cov(O, dU/da) = <O * dU/da> - <O><dU/da>
        E_dU_da = (dU_da * prob).sum() * self.dx
        E_O_dU_da = (O * dU_da * prob).sum() * self.dx

        cov = E_O_dU_da - E_O * E_dU_da

        # REINFORCE Identity: d<O>/da = -beta * Cov(O, dU/da)
        grad = -self.beta * cov

        return grad.item()

# -----------------------------------------------------------------------------
# Experiment Runners
# -----------------------------------------------------------------------------

def run_trial(system_type, regime_steps, method_type):
    """Run a single gradient estimation trial."""

    # Setup System
    torch.manual_seed(int(time.time() * 1000) % 100000)

    if system_type == "Langevin":
        # Non-chaotic, stochastic
        potential = DoubleWell(barrier_height=BARRIER_HEIGHT)
        integrator = OverdampedLangevin(gamma=1.0, kT=1.0/BETA)
        is_stochastic = True
    elif system_type == "NoseHoover":
        # Chaotic, deterministic (mostly)
        potential = DoubleWell(barrier_height=BARRIER_HEIGHT)
        # Use Checkpointed integrator for Adjoint, regular for others
        if method_type == "Adjoint":
            integrator = CheckpointedNoseHoover(kT=1.0/BETA, mass=1.0, Q=1.0)
        else:
            integrator = NoseHoover(kT=1.0/BETA, mass=1.0, Q=1.0)
        is_stochastic = False
    else:
        raise ValueError(f"Unknown system: {system_type}")

    # Initial State
    x0 = torch.randn(1, 1) # Single particle, 1D
    v0 = torch.randn(1, 1) if not is_stochastic else None

    start_time = time.time()

    grad_val = None

    # Run Method
    if method_type == "BPTT":
        # Standard Backprop
        if is_stochastic:
            traj = integrator.run(x0, potential.force, dt=DT, n_steps=regime_steps)
        else:
            traj_x, traj_v = integrator.run(x0, v0, potential.force, dt=DT, n_steps=regime_steps)
            traj = traj_x

        burn_in = regime_steps // 2
        samples = traj[burn_in:]
        loss = (samples**2).mean()

        integrator.zero_grad()
        potential.zero_grad()

        loss.backward()
        grad_val = potential.barrier_height.grad.item()

    elif method_type == "REINFORCE":
        # Score Function Estimator
        with torch.no_grad():
            if is_stochastic:
                traj = integrator.run(x0, potential.force, dt=DT, n_steps=regime_steps)
            else:
                traj_x, _ = integrator.run(x0, v0, potential.force, dt=DT, n_steps=regime_steps)
                traj = traj_x

        burn_in = regime_steps // 2
        samples = traj[burn_in:]

        estimator = ReinforceEstimator(potential, beta=BETA)
        grads = estimator.estimate_gradient(samples, observable=lambda x: x**2)
        if 'barrier_height' in grads:
            grad_val = grads['barrier_height'].item()
        else:
            grad_val = 0.0

    elif method_type == "Girsanov":
        # Only valid for Langevin
        if system_type != "Langevin":
            return None, 0

        with torch.no_grad():
             traj = integrator.run(x0, potential.force, dt=DT, n_steps=regime_steps)

        sigma = (2 * (1.0/BETA) / 1.0)**0.5 # sqrt(2kT/gamma)
        estimator = GirsanovEstimator(potential, sigma=sigma, beta=BETA)

        grads = estimator.estimate_gradient(traj, observable=lambda x: (x**2).mean(), dt=DT)
        if 'barrier_height' in grads:
            grad_val = grads['barrier_height'].item()
        else:
            grad_val = 0.0

    elif method_type == "Adjoint":
        # Adjoint does not support potential parameters yet
        return None, 0

    elapsed = time.time() - start_time
    return grad_val, elapsed


# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

def main():
    print("Variance-Bias-Chaos Trade-off Exploration")
    print("-----------------------------------------")

    # 1. Compute Ground Truth
    gt_calculator = GroundTruth(BARRIER_HEIGHT, BETA)
    true_grad = gt_calculator.compute_gradient()
    print(f"True Gradient (d<x^2>/da): {true_grad:.6f}\n")

    results = []

    # 2. Run Experiments
    methods = ["BPTT", "REINFORCE", "Girsanov", "Adjoint"]
    systems = ["Langevin", "NoseHoover"]

    total_configs = len(systems) * len(REGIMES) * len(methods)
    pbar = 0

    for sys_name in systems:
        for reg_name, steps in REGIMES.items():
            for method in methods:
                pbar += 1

                # Skip invalid combinations
                if method == "Girsanov" and sys_name != "Langevin":
                    continue
                if method == "Adjoint" and sys_name != "NoseHoover":
                    continue

                print(f"[{pbar}/{total_configs}] Running {sys_name} | {reg_name} | {method} ...", end="", flush=True)

                grads = []
                times = []

                for _ in range(N_TRIALS):
                    g, t = run_trial(sys_name, steps, method)
                    if g is not None:
                        grads.append(g)
                        times.append(t)

                if not grads:
                    print(" N/A")
                    continue

                grads = np.array(grads)
                bias = grads.mean() - true_grad
                variance = grads.var()
                mse = bias**2 + variance
                avg_time = np.mean(times)

                print(f" MSE={mse:.2e} Time={avg_time:.3f}s")

                results.append({
                    "System": sys_name,
                    "Regime": reg_name,
                    "Config": f"{sys_name}\n{reg_name}",
                    "Method": method,
                    "Bias": np.abs(bias),
                    "Variance": variance,
                    "MSE": mse,
                    "Time": avg_time
                })

    # 3. Visualization
    df = pd.DataFrame(results)
    apply_style()

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # --- Plot 1: MSE Landscape (Heatmap) ---
    ax_heatmap = fig.add_subplot(gs[0, :])

    # Pivot data for heatmap: Rows=Method, Cols=Config
    # We want valid configs only
    heatmap_data = df.pivot(index="Method", columns="Config", values="MSE")

    # Reorder columns and rows if desired
    desired_methods = ["BPTT", "REINFORCE", "Girsanov", "Adjoint"]
    heatmap_data = heatmap_data.reindex(desired_methods)

    # Use Log10(MSE) for better color scaling
    log_mse = np.log10(heatmap_data.astype(float))

    sns.heatmap(log_mse, annot=True, fmt=".1f", cmap="magma_r",
                cbar_kws={'label': r'$\log_{10}(\mathrm{MSE})$'}, ax=ax_heatmap)
    ax_heatmap.set_title("Gradient Estimation Error Landscape", fontsize=14)
    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("")

    # --- Plot 2: Cost-Accuracy Trade-off (Scatter) ---
    ax_scatter = fig.add_subplot(gs[1, :])

    # Color palette
    palette = {m: COLORS.get(m.lower(), "#333333") for m in methods}
    palette["BPTT"] = COLORS["bptt"]
    palette["REINFORCE"] = COLORS["reinforce"]
    palette["Girsanov"] = COLORS["girsanov"]
    palette["Adjoint"] = COLORS["adjoint"]

    # Plot
    sns.scatterplot(data=df, x="Time", y="MSE", hue="Method", style="Config",
                    s=150, alpha=0.9, palette=palette, ax=ax_scatter)

    ax_scatter.set_yscale("log")
    ax_scatter.set_xscale("log")
    ax_scatter.set_title("Cost-Accuracy Trade-off (Pareto Frontier)", fontsize=14)
    ax_scatter.set_ylabel("MSE (Log Scale)")
    ax_scatter.set_xlabel("Wall-clock Time (s, Log Scale)")
    ax_scatter.grid(True, which="both", ls="--", alpha=0.2)

    # Annotate points
    for i, row in df.iterrows():
        ax_scatter.text(row["Time"]*1.1, row["MSE"], row["Config"].replace('\n', ' '),
                        fontsize=8, alpha=0.7)

    plt.tight_layout()

    # Save
    assets_dir = get_assets_dir()
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, "variance_bias_tradeoff.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()
