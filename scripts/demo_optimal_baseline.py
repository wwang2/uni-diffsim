"""
Demo: REINFORCE Baselines Comparison
===================================

This script demonstrates the impact of different baselines on the variance of the
REINFORCE gradient estimator for equilibrium sampling.

We estimate the gradient: ∇_θ ⟨O⟩ = -β Cov(O, ∇_θ U)

Estimators compared:
1. Zero Baseline: b=0 (Biased because E[∇U] != 0)
2. Standard (Mean) Baseline: b=mean(O) (Unbiased, standard implementation)
3. Oracle Baseline: b=E[O] (Unbiased, lowest variance, requires knowing true mean)

The "Standard" baseline corresponds to the `optimal` implementation in this codebase,
as it is the optimal choice available without external knowledge.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uni_diffsim.potentials import Harmonic
from uni_diffsim.gradient_estimators import ReinforceEstimator
from uni_diffsim.plotting import apply_style, COLORS

def run_demo():
    apply_style()
    print("REINFORCE Baseline Comparison Demo")
    print("----------------------------------")

    # Setup: Harmonic oscillator
    torch.manual_seed(42)
    np.random.seed(42)

    kT = 1.0
    beta = 1.0 / kT
    k_true = 2.0
    potential = Harmonic(k=k_true)

    # Theory
    # U = 0.5 * k * x^2
    # ∇_k U = 0.5 * x^2
    # O = x^2
    # E[O] = 1/(beta*k) = 0.5
    # Target Grad = -0.25
    theory_grad = -0.25
    true_mean_O = 0.5

    print(f"Target Gradient: {theory_grad:.4f}")

    # Generate large pool of samples
    sigma = (1.0 / (beta * k_true))**0.5
    n_total = 10000
    samples_pool = torch.randn(n_total, 1) * sigma

    def observable(x):
        return (x**2).sum(dim=-1)

    # Create estimators
    est_zero = ReinforceEstimator(potential, beta=beta, baseline=torch.tensor(0.0))
    est_std = ReinforceEstimator(potential, beta=beta, baseline=None)
    est_oracle = ReinforceEstimator(potential, beta=beta, baseline=torch.tensor(true_mean_O))

    # Bootstrap experiment
    n_batches = 2000
    batch_size = 50

    grads_zero = []
    grads_std = []
    grads_oracle = []

    print(f"\nRunning {n_batches} estimates with batch size {batch_size}...")

    for i in range(n_batches):
        idx = torch.randint(0, n_total, (batch_size,))
        batch = samples_pool[idx]

        grads_zero.append(est_zero.estimate_gradient(batch, observable)['k'].item())
        grads_std.append(est_std.estimate_gradient(batch, observable)['k'].item())
        grads_oracle.append(est_oracle.estimate_gradient(batch, observable)['k'].item())

    # Analysis & Plotting
    plt.figure(figsize=(10, 6))

    # Plot distributions
    sns.kdeplot(grads_zero, fill=True, label="Zero Baseline (b=0)", color=COLORS['red'], alpha=0.3)
    sns.kdeplot(grads_std, fill=True, label="Standard Baseline (b=mean(O))", color=COLORS['reinforce'], alpha=0.3)
    sns.kdeplot(grads_oracle, fill=True, label="Oracle Baseline (b=E[O])", color=COLORS['green'], alpha=0.3)

    # Plot theoretical mean
    plt.axvline(theory_grad, color=COLORS['theory'], linestyle='--', linewidth=2, label="True Gradient")

    plt.title("Distribution of REINFORCE Gradient Estimates")
    plt.xlabel(f"Gradient Estimate (Target: {theory_grad})")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    save_path = "demo_optimal_baseline.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")

    # Text Stats
    def stats(name, grads):
        grads = np.array(grads)
        mean = np.mean(grads)
        var = np.var(grads)
        bias = mean - theory_grad
        mse = np.mean((grads - theory_grad)**2)
        print(f"\n{name}:")
        print(f"  Mean: {mean:.4f} (Bias: {bias:.4f})")
        print(f"  Var:  {var:.4f}")
        print(f"  MSE:  {mse:.4f}")
        return var

    print("\nStatistical Summary:")
    stats("Zero Baseline", grads_zero)
    var_std = stats("Standard Baseline", grads_std)
    var_orc = stats("Oracle Baseline", grads_oracle)

    print(f"\nVariance Reduction (Standard vs Oracle): {100*(var_std - var_orc)/var_std:.1f}%")
    print("Conclusion: Standard Baseline removes bias effectively compared to Zero Baseline.")

if __name__ == "__main__":
    run_demo()
