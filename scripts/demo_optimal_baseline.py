"""
Demo: Optimal Baseline for REINFORCE Gradient Estimator
======================================================

This script demonstrates the behavior of REINFORCE baselines.

Theory:
The gradient of <O> is -beta * Cov(O, dU/dtheta).
Standard estimator: -beta * [ mean(O * dU) - mean(O) * mean(dU) ].
This corresponds to using a baseline b = mean(O).

Is there an "optimal" constant baseline b*?
The variance of the estimator `mean( (O - b) * dU )` is minimized by:
b* = E[ O * dU^2 ] / E[ dU^2 ].

However, unless b = E[O], the estimator `mean( (O - b) * dU )` is BIASED because E[dU] != 0.
Bias = -beta * (E[O] - b) * E[dU].

Since we require an unbiased estimator, we must use b = E[O].
Since E[O] is unknown, we use b = mean(O) (which yields the sample covariance).

This demo compares:
1. Standard REINFORCE (b = sample mean of O).
2. "Optimal" Variance Minimizing Baseline (b = b*). Note: This is BIASED.
3. Oracle Baseline (b = True E[O]). This is unbiased and has lower variance than sample mean.

"""

import torch
from uni_diffsim.potentials import Harmonic
from uni_diffsim.gradient_estimators import ReinforceEstimator
import numpy as np

def run_demo():
    print("REINFORCE Baseline Comparison")
    print("-----------------------------")

    # Setup
    torch.manual_seed(42)
    kT = 1.0
    beta = 1.0 / kT
    k_true = 2.0

    potential = Harmonic(k=k_true)

    # Theory
    # O(x) = x^2. <O> = 1/(beta*k) = 0.5.
    # d<O>/dk = -0.25.
    theory_grad = -0.25
    true_mean_O = 0.5
    print(f"Theoretical Gradient: {theory_grad:.6f}")

    # Generate samples
    sigma = (1.0 / (beta * k_true))**0.5
    n_total = 10000
    samples_pool = torch.randn(n_total, 1) * sigma

    def observable(x):
        return (x**2).sum(dim=-1)

    # Estimators
    # 1. Standard (b = mean(O))
    est_std = ReinforceEstimator(potential, beta=beta, baseline=None)

    # 2. Optimal Variance (Biased) - We simulate this manually or using the current 'optimal' impl
    est_opt = ReinforceEstimator(potential, beta=beta, baseline="optimal")

    # 3. Oracle (b = 0.5)
    est_oracle = ReinforceEstimator(potential, beta=beta, baseline=torch.tensor(true_mean_O))

    # Bootstrap
    n_batches = 2000
    batch_size = 50

    grads_std = []
    grads_opt = []
    grads_oracle = []

    print(f"\nRunning {n_batches} estimates with batch size {batch_size}...")

    for i in range(n_batches):
        idx = torch.randint(0, n_total, (batch_size,))
        batch = samples_pool[idx]

        g_std = est_std.estimate_gradient(batch, observable)['k'].item()
        g_opt = est_opt.estimate_gradient(batch, observable)['k'].item()
        g_oracle = est_oracle.estimate_gradient(batch, observable)['k'].item()

        grads_std.append(g_std)
        grads_opt.append(g_opt)
        grads_oracle.append(g_oracle)

    # Analysis
    def analyze(name, grads):
        grads = np.array(grads)
        mean = np.mean(grads)
        bias = mean - theory_grad
        var = np.var(grads)
        mse = np.mean((grads - theory_grad)**2)
        print(f"\n{name}:")
        print(f"  Mean: {mean:.6f} (Bias: {bias:.6f})")
        print(f"  Var:  {var:.6f}")
        print(f"  MSE:  {mse:.6f}")
        return var, mse

    v_std, mse_std = analyze("Standard (Sample Mean)", grads_std)
    v_opt, mse_opt = analyze("Optimal Variance (Biased)", grads_opt)
    v_orc, mse_orc = analyze("Oracle (True Mean)", grads_oracle)

    print("\nComparison:")
    print(f"Oracle vs Standard Variance Reduction: {(v_std - v_orc)/v_std*100:.1f}%")
    print(f"Optimal vs Standard MSE Reduction:     {(mse_std - mse_opt)/mse_std*100:.1f}%")

    if abs(np.mean(grads_opt) - theory_grad) > 0.1:
        print("\nNOTE: The 'Optimal' baseline (ratio of expectations) introduces significant bias for energy gradients.")
        print("This confirms that it should not be used for unbiased gradient estimation.")

if __name__ == "__main__":
    run_demo()
