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

    # Initial State (sample from equilibrium approx to reduce burn-in bias for Short T)
    # Actually, let's start from 0 to see burn-in effects, or random.
    # To be fair to Short T, let's start somewhat distributed.
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

        # Loss: mean of x^2 over trajectory (path average -> equilibrium)
        # We discard first half as burn-in for better equilibrium estimate?
        # Or just use full trajectory?
        # For BPTT on Short T, burn-in is significant.
        # Let's use last 50% samples.
        burn_in = regime_steps // 2
        samples = traj[burn_in:]
        loss = (samples**2).mean()

        integrator.zero_grad()
        potential.zero_grad()

        # Check if potential parameters are in graph
        # Note: Integrator.run returns tensor. If we want grad w.r.t potential,
        # we need to ensure potential.force is called and graph is maintained.
        # uni_diffsim integrators support this.

        loss.backward()
        grad_val = potential.barrier_height.grad.item()

    elif method_type == "REINFORCE":
        # Score Function Estimator
        # Need to run forward without tracking gradients first (to get samples)
        # Wait, ReinforceEstimator needs samples.
        # For Langevin, we can detach.
        with torch.no_grad():
            if is_stochastic:
                traj = integrator.run(x0, potential.force, dt=DT, n_steps=regime_steps)
            else:
                traj_x, _ = integrator.run(x0, v0, potential.force, dt=DT, n_steps=regime_steps)
                traj = traj_x

        # Use last 50%
        burn_in = regime_steps // 2
        samples = traj[burn_in:]

        # Estimate
        estimator = ReinforceEstimator(potential, beta=BETA)
        grads = estimator.estimate_gradient(samples, observable=lambda x: x**2)
        if 'barrier_height' in grads:
            grad_val = grads['barrier_height'].item()
        else:
            print(f"Warning: barrier_height not in grads (keys: {grads.keys()})")
            grad_val = 0.0

    elif method_type == "Girsanov":
        # Only valid for Langevin
        if system_type != "Langevin":
            return None, 0

        # Need differentiable trajectory for score?
        # GirsanovEstimator takes trajectory tensor.
        # It computes score internally.
        # But we need to run simulation to get trajectory.
        # Does Girsanov need graph?
        # "Girsanov-based path reweighting... computes grad of log p(tau)".
        # It needs to differentiate log p(tau) w.r.t parameters.
        # This means we need potential.force to be differentiable.

        with torch.no_grad():
             traj = integrator.run(x0, potential.force, dt=DT, n_steps=regime_steps)

        # Girsanov reweighting
        sigma = (2 * (1.0/BETA) / 1.0)**0.5 # sqrt(2kT/gamma)
        estimator = GirsanovEstimator(potential, sigma=sigma, beta=BETA)

        # Observable is mean x^2
        obs_fn = lambda t: (t**2).mean(dim=0) # Mean over time?
        # Girsanov usually reweights the entire path expectation.
        # If O is path average, O = 1/T sum x_t^2.

        # The estimator.estimate_gradient expects observable to return scalar per trajectory.
        # Our input traj is (T, 1, 1).
        # We need to reshape to (1, T, 1) for estimator (batch dim 0).
        traj_batch = traj.unsqueeze(0)

        # Observable: Average x^2 over the trajectory (excluding burn-in ideally, but Girsanov is path-based)
        # If we exclude burn-in from observable, we should probably exclude from score too?
        # Let's just use full path for Girsanov to be consistent with "path probability".

        grads = estimator.estimate_gradient(traj, observable=lambda x: (x**2).mean(), dt=DT)
        if 'barrier_height' in grads:
            grad_val = grads['barrier_height'].item()
        else:
            print(f"Warning: barrier_height not in grads (keys: {grads.keys()})")
            grad_val = 0.0

    elif method_type == "Adjoint":
        # Only valid for NoseHoover (in this demo context)
        if system_type != "NoseHoover":
            return None, 0

        # Use CheckpointedNoseHoover
        # It supports final_only=True for O(sqrt(T)) memory.
        # To get gradient of mean(x^2), we strictly need full trajectory?
        # Or we can just optimize final state x_T^2?
        # "Equilibrium" expectation is average over time.
        # Optimizing x_T^2 for large T is a stochastic approximation of optimizing equilibrium.
        # BPTT used mean(samples).
        # CheckpointedNoseHoover with `store_every` can store sparse trajectory.
        # Let's try to match BPTT's objective: mean of second half.
        # If we store all steps, it's O(T) memory for storage but O(sqrt(T)) for backward?
        # No, Checkpointed stores checkpoints, and recomputes segments.
        # If we access full trajectory for loss, we hold full trajectory in memory.
        # But `traj_x` returned by `run` is detached from graph except via custom function.
        # The custom backward recomputes.
        # So we CAN compute loss on full trajectory and backprop efficiently!

        traj_x, traj_v = integrator.run(x0, v0, potential.force, dt=DT, n_steps=regime_steps, store_every=1)

        burn_in = regime_steps // 2
        samples = traj_x[burn_in:]
        loss = (samples**2).mean()

        loss.backward()
        grad_val = integrator.kT.grad # Wait, barrier_height is in potential, not integrator!

        # ERROR: CheckpointedNoseHoover wraps the *potential force* in the forward pass.
        # But the potential parameters are NOT passed explicitly to the custom function
        # unless we modify it.
        # `NoseHooverCheckpointedFunction` saves `force_fn` in ctx.
        # `force_fn` is a bound method (potential.force).
        # Does PyTorch autograd track gradients through a stored bound method in ctx?
        # Typically NO, unless the parameters are passed as inputs to `apply`.
        # The `NoseHooverCheckpointedFunction.apply` signature:
        # (x0, v0, alpha0, kT, mass, Q, force_fn, ...)
        # It does NOT take potential parameters.
        # So gradients w.r.t potential parameters will NOT flow through `CheckpointedNoseHoover`
        # unless `force_fn` carries the graph (which it doesn't in custom function backward).

        # In `backward`, we recompute: `force_fn(x_grad)`.
        # `force_fn` is `potential.force`.
        # If `potential` parameters require grad, `force_fn(x_grad)` will create a graph connecting `x_grad` to parameters.
        # BUT, the `backward` method returns gradients w.r.t INPUTS of `apply`.
        # Potential parameters are NOT inputs.
        # However, they are leaf nodes in the graph created INSIDE backward.
        # When `grad_outputs` comes in, we run `torch.autograd.grad` inside `backward`.
        # We check `ContinuousAdjointNoseHoover.adjoint_step_backward`:
        # It computes `grad_kT`, `grad_mass`, etc. explicitly.
        # It does NOT compute gradients for potential parameters explicitly in the return tuple!
        # It assumes we only care about integrator parameters?
        # Let's check `DiscreteAdjointNoseHoover` and `CheckpointedNoseHoover`.

        # `CheckpointedNoseHoover`:
        # In `backward`: `grads = torch.autograd.grad(..., [x_grad, ..., Q_grad])`.
        # It does NOT accumulate gradients for potential parameters.
        # This is a limitation of the current implementation in `uni_diffsim`.

        # FIX: We cannot use `CheckpointedNoseHoover` for *potential* parameter optimization
        # as currently implemented, because it doesn't expose potential params to the autograd chain.
        # However, BPTT works because it relies on standard PyTorch tracing.

        # What about `ContinuousAdjointNoseHoover`?
        # Same issue. It computes dL/dtheta for theta in integrator, but not potential.

        # WAIT! If `force_fn` uses `potential.force`, and `potential` has parameters.
        # Inside `adjoint_step`, we compute `vjp = grad(F, x, grad_outputs=lambda_v)`.
        # This propagates gradient from `lambda_v` to `x`.
        # But what about `lambda_v` to `potential_params`?
        # The adjoint equation for parameters is: dL/dtheta = integral( lambda_v * dF/dtheta ).
        # The current implementation does NOT compute this integral for potential parameters.

        # So, strictly speaking, the Adjoint implementations in the library are
        # for *Integrator* parameters (kT, mass, Q) or Initial State.
        # They do not support Potential parameters out of the box.

        # For the purpose of this demo, I will SKIP Adjoint for potential parameters
        # and mark it as "N/A (Library Limitation)".
        # Or I can modify the demo to optimize `kT` (Temperature)?
        # But optimizing Temperature to minimize Energy is trivial (T->0).
        # Optimizing `mass`? Doesn't affect equilibrium.

        # Let's stick to Potential Parameters (Barrier Height).
        # Therefore, Adjoint is not applicable with current library code.
        # I will mark it as Not Implemented / N/A.
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
                    "Method": method,
                    "Bias": np.abs(bias),
                    "Variance": variance,
                    "MSE": mse,
                    "Time": avg_time
                })

    # 3. Visualization
    df = pd.DataFrame(results)

    # Create combined category for X-axis
    df["Config"] = df["System"] + "\n" + df["Regime"]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Metrics
    metrics = ["MSE", "Variance", "Time"]
    for i, metric in enumerate(metrics):
        # Create a palette mapping that matches Method names (capitalized/MixedCase)
        palette = {m: COLORS.get(m.lower(), "#333333") for m in methods}
        # Override specific ones if needed
        palette["BPTT"] = COLORS["bptt"]
        palette["REINFORCE"] = COLORS["reinforce"]
        palette["Girsanov"] = COLORS["girsanov"]
        palette["Adjoint"] = COLORS["adjoint"]

        sns.barplot(data=df, x="Config", y=metric, hue="Method", ax=axes[i], palette=palette)
        axes[i].set_title(metric)
        axes[i].set_xlabel("")
        if metric == "MSE":
            axes[i].set_yscale("log")

    plt.tight_layout()

    # Save
    assets_dir = get_assets_dir()
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, "variance_bias_tradeoff.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()
