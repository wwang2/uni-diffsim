"""
Demo: REINFORCE Baselines Comparison
===================================

This script demonstrates the impact of different baselines on the variance of the
REINFORCE gradient estimator for equilibrium sampling.

We estimate the gradient: ∇_θ ⟨O⟩ = -β Cov(O, ∇_θ U)

Estimators:
1. No Baseline: -β * mean( (O - 0) * ∇U ) - (Bias correction term needed if E[∇U]!=0)
   Wait, if we don't use a baseline (b=0), we compute -β * mean( O * ∇U ).
   This estimates -β * E[O * ∇U].
   But the target is -β * Cov(O, ∇U) = -β * ( E[O * ∇U] - E[O]E[∇U] ).
   So "No Baseline" (estimator of expectation) is WRONG for the gradient of free energy
   unless we explicitly subtract mean(O)*mean(∇U).

   However, the ReinforceEstimator class implements the Covariance formula directly
   if we use accumulate/get_gradient.

   In `estimate_gradient(reduce=True)`:
   It computes -β * mean( (O - b) * ∇U ).
   For this to be consistent, we must have E[ (O - b) * ∇U ] = Cov(O, ∇U).
   E[O∇U] - b E[∇U] = E[O∇U] - E[O]E[∇U].
   So b MUST be E[O].

   If we use b = 0, we get -β * E[O∇U]. This is BIASED by E[O]E[∇U].

   So "No Baseline" (b=0) is biased.
   "Mean Baseline" (b=mean(O)) estimates Covariance.
   "Oracle Baseline" (b=E[O]) estimates Covariance with lower variance.

   This demo compares:
   1. "Zero Baseline": b=0 (Biased)
   2. "Standard/Mean Baseline": b=mean(O) (Unbiased*)
   3. "Oracle Baseline": b=E[O] (Unbiased, lower variance)

   *Unbiased asymptotically / MVUE for covariance.
"""

import torch
from uni_diffsim.potentials import Harmonic
from uni_diffsim.gradient_estimators import ReinforceEstimator
import numpy as np

def run_demo():
    print("REINFORCE Baseline Comparison Demo")
    print("----------------------------------")

    # Setup: Harmonic oscillator
    torch.manual_seed(42)
    kT = 1.0
    beta = 1.0 / kT
    k_true = 2.0

    potential = Harmonic(k=k_true)

    # Theory
    # U = 0.5 * k * x^2
    # ∇_k U = 0.5 * x^2
    # O = x^2
    # E[O] = 1/(beta*k) = 0.5
    # E[∇U] = 0.5 * E[O] = 0.25
    # Cov(O, ∇U) = Cov(O, 0.5 O) = 0.5 Var(O)
    # Var(O) = 2/(beta*k)^2 = 2 * (0.5)^2 = 0.5  (For Gamma/Chi-sq)
    # Target Grad = -beta * Cov = -1.0 * 0.5 * 0.5 = -0.25.

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
    # 1. Zero Baseline (b=0)
    est_zero = ReinforceEstimator(potential, beta=beta, baseline=torch.tensor(0.0))

    # 2. Standard (b=mean(O)) - This is the default
    est_std = ReinforceEstimator(potential, beta=beta, baseline=None)

    # 3. Oracle (b=E[O]=0.5)
    est_oracle = ReinforceEstimator(potential, beta=beta, baseline=torch.tensor(true_mean_O))

    # Bootstrap experiment
    n_batches = 2000
    batch_size = 50

    grads_zero = []
    grads_std = []
    grads_oracle = []

    for i in range(n_batches):
        idx = torch.randint(0, n_total, (batch_size,))
        batch = samples_pool[idx]

        # We need to be careful: estimate_gradient returns -beta * mean((O-b)*g)
        # For zero baseline, this is -beta * mean(O*g).
        # This is NOT the gradient of free energy. It includes the E[O]E[g] term.
        # But let's show that it is indeed biased/wrong.

        g_zero = est_zero.estimate_gradient(batch, observable)['k'].item()
        g_std = est_std.estimate_gradient(batch, observable)['k'].item()
        g_oracle = est_oracle.estimate_gradient(batch, observable)['k'].item()

        grads_zero.append(g_zero)
        grads_std.append(g_std)
        grads_oracle.append(g_oracle)

    # Analysis
    def stats(name, grads):
        grads = np.array(grads)
        mean = np.mean(grads)
        var = np.var(grads)
        mse = np.mean((grads - theory_grad)**2)
        print(f"\n{name}:")
        print(f"  Mean Estimate: {mean:.4f}")
        print(f"  Bias:          {mean - theory_grad:.4f}")
        print(f"  Variance:      {var:.4f}")
        print(f"  MSE:           {mse:.4f}")
        return var

    print("\nResults (Batch Size = 50):")
    stats("Zero Baseline (b=0)", grads_zero)
    var_std = stats("Standard Baseline (b=mean(O))", grads_std)
    var_orc = stats("Oracle Baseline (b=E[O])", grads_oracle)

    print("\nAnalysis:")
    print("1. Zero Baseline is heavily biased because E[∇U] ≠ 0.")
    print("   (It estimates -β E[O∇U] instead of -β Cov(O, ∇U))")
    print("2. Standard Baseline (Mean) removes the bias effectively.")
    print("3. Oracle Baseline reduces variance further if E[O] is known.")

    print(f"\nVariance Reduction (Standard vs Oracle): {100*(var_std - var_orc)/var_std:.1f}%")
    print("The 'Standard' baseline is the optimal choice when E[O] is unknown.")

if __name__ == "__main__":
    run_demo()
