"""Gradient estimators for differentiable simulations.

This module provides gradient computation methods that bypass trajectory differentiation,
using statistical estimators instead.

Gradient Estimators:
- ReinforceEstimator: REINFORCE/score-function estimator for equilibrium observables
- reinforce_gradient: Functional API for REINFORCE gradient computation

Theory:
For a potential U(x; θ) with equilibrium distribution ρ(x|θ) ∝ exp(-βU(x;θ)),
the gradient of an observable expectation is:

    ∇_θ ⟨O⟩ = -β Cov_ρ(O, ∇_θU) = -β [⟨O ∇_θU⟩ - ⟨O⟩⟨∇_θU⟩]

This is the REINFORCE/TPT (Thermodynamic Perturbation Theory) estimator.

Advantages over BPTT:
- No gradient explosion in chaotic systems
- Works with non-differentiable dynamics
- O(1) memory vs O(T) for full trajectory storage

Disadvantages:
- Higher variance than reparameterization
- Requires equilibrium samples (burn-in)
- Only works for equilibrium observables
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional, Union

# Type aliases
Observable = Callable[[torch.Tensor], torch.Tensor]
PotentialFunc = Callable[[torch.Tensor], torch.Tensor]


class ReinforceEstimator(nn.Module):
    """REINFORCE gradient estimator for equilibrium observables.

    Computes gradients of equilibrium expectations using the identity:
        ∇_θ ⟨O⟩ = -β [⟨O ∇_θU⟩ - ⟨O⟩⟨∇_θU⟩]

    This avoids backpropagating through dynamics, making it suitable for:
    - Chaotic systems where BPTT fails
    - Long trajectories where memory is limited
    - Non-differentiable integrators

    The estimator can be used in two modes:
    1. Stateless: Call estimate_gradient() with samples
    2. Accumulating: Call accumulate() multiple times, then get_gradient()

    Example:
        >>> potential = DoubleWell(barrier_height=1.0)
        >>> estimator = ReinforceEstimator(potential, beta=1.0)
        >>>
        >>> # Run dynamics to get equilibrium samples
        >>> samples = integrator.run(x0, potential.force, dt, n_steps)[burn_in:]
        >>>
        >>> # Compute gradient of mean position
        >>> grad = estimator.estimate_gradient(samples, observable=lambda x: x.mean())
        >>>
        >>> # Apply gradient to potential parameters
        >>> for name, param in potential.named_parameters():
        >>>     param.grad = grad[name]

    Args:
        potential: Potential energy function (nn.Module with parameters).
        beta: Inverse temperature 1/kT. Higher beta = lower temperature.
        baseline: Optional control variate for variance reduction.
            Can be:
            - None: Use mean observable as baseline (default)
            - "optimal": Compute optimal baseline (more expensive)
            - Callable: Custom baseline function b(x)
            - Tensor: Constant baseline value
    """

    def __init__(
        self,
        potential: nn.Module,
        beta: float = 1.0,
        baseline: Optional[Union[str, Callable, torch.Tensor]] = None,
    ):
        super().__init__()
        self.potential = potential
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.baseline = baseline

        # Accumulator state for batched gradient estimation
        self._reset_accumulators()

    def _reset_accumulators(self):
        """Reset internal accumulators for batched estimation."""
        self._n_samples = 0
        self._sum_O = None
        self._sum_O_dU = {}  # param_name -> accumulated O * dU/dθ
        self._sum_dU = {}    # param_name -> accumulated dU/dθ

    def _compute_dU_dtheta(
        self,
        x: torch.Tensor,
        create_graph: bool = False
    ) -> dict[str, torch.Tensor]:
        """Compute gradient of U w.r.t. potential parameters.

        Args:
            x: Positions (..., dim)
            create_graph: Whether to create graph for higher-order derivatives

        Returns:
            Dictionary mapping parameter names to their gradients.
            Each gradient has shape (n_samples, *param_shape).
        """
        # Flatten batch dimensions for easier handling
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # (n_samples, dim)
        n_samples = x_flat.shape[0]

        # Compute energy for each sample
        U = self.potential.energy(x_flat)  # (n_samples,)

        # Compute gradients w.r.t. each parameter
        grads = {}
        for name, param in self.potential.named_parameters():
            if param.requires_grad:
                # Compute per-sample gradient
                # grad shape: (n_samples, *param_shape)
                grad = torch.autograd.grad(
                    U.sum(),
                    param,
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                if grad is not None:
                    # Expand to per-sample if needed
                    if grad.shape == param.shape:
                        # Same gradient for all samples - this is common
                        # We need to weight by sample later
                        grads[name] = grad.unsqueeze(0).expand(n_samples, *param.shape)
                    else:
                        grads[name] = grad

        return grads

    def _compute_per_sample_dU(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute per-sample energy and gradients using vmap-style computation.

        For proper REINFORCE, we need per-sample gradients:
        dU_i/dθ for each sample i.

        Args:
            x: Positions (n_samples, dim)

        Returns:
            U: Energy per sample (n_samples,)
            grads: Dictionary of per-sample gradients (n_samples, *param_shape)
        """
        n_samples = x.shape[0]

        # Option 1: Use functorch vmap if available (PyTorch >= 2.0)
        # Option 2: Loop over samples (slower but always works)

        # For now, use a memory-efficient approach:
        # Compute gradient of sum(U), which gives sum of per-sample gradients
        # This works because REINFORCE only needs covariance estimates

        U = self.potential.energy(x)  # (n_samples,)

        grads = {}
        for name, param in self.potential.named_parameters():
            if param.requires_grad:
                # For parameters that enter linearly into U,
                # dU/dθ is the same for all samples, just scaled
                # For general case, we approximate using batch gradient
                grad = torch.autograd.grad(
                    U.sum(),
                    param,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                if grad is not None:
                    grads[name] = grad

        return U, grads

    def estimate_gradient(
        self,
        samples: torch.Tensor,
        observable: Optional[Observable] = None,
        reduce: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Estimate gradient of ⟨O⟩ w.r.t. potential parameters.

        Uses the REINFORCE identity:
            ∇_θ ⟨O⟩ = -β Cov(O, ∇_θU) = -β [⟨O ∇_θU⟩ - ⟨O⟩⟨∇_θU⟩]

        Args:
            samples: Equilibrium samples (n_samples, ..., dim) or trajectory
                     (n_steps, ..., dim). Will be flattened to (n_samples, dim).
            observable: Function O(x) -> scalar or (n_samples,) tensor.
                       If None, uses mean energy as observable.
            reduce: If True, return mean gradient. If False, return per-sample.

        Returns:
            Dictionary mapping parameter names to gradient estimates.

        Note:
            For accurate gradients, samples should be from equilibrium.
            Include burn-in period before calling this function.
        """
        # Flatten trajectory to samples
        original_shape = samples.shape
        samples_flat = samples.reshape(-1, samples.shape[-1])  # (n_samples, dim)
        n_samples = samples_flat.shape[0]

        # Compute observable values
        if observable is None:
            O = self.potential.energy(samples_flat)  # Default: energy
        else:
            O = observable(samples_flat)

        # Handle different observable output shapes
        if O.dim() == 0:
            O = O.expand(n_samples)
        elif O.shape[0] != n_samples:
            raise ValueError(f"Observable must return (n_samples,) tensor, got {O.shape}")

        # Compute ⟨O⟩
        O_mean = O.mean()

        # Apply baseline for variance reduction
        if self.baseline is None:
            # Default: subtract mean (standard REINFORCE)
            O_centered = O - O_mean
        elif self.baseline == "optimal":
            # Optimal baseline requires second pass - compute later
            O_centered = O - O_mean  # Placeholder
        elif callable(self.baseline):
            b = self.baseline(samples_flat)
            O_centered = O - b
        elif isinstance(self.baseline, torch.Tensor):
            O_centered = O - self.baseline
        else:
            O_centered = O - O_mean

        # Compute per-sample gradients dU_i/dθ
        # We need: ⟨O_i * dU_i/dθ⟩ - ⟨O⟩ * ⟨dU/dθ⟩

        # For efficiency, we use the fact that for most potentials,
        # dU/dθ factors as: dU/dθ = f(θ) * g(x)
        # So ⟨O * dU/dθ⟩ = f(θ) * ⟨O * g(x)⟩

        grads = {}

        # Compute energy and its gradients
        U = self.potential.energy(samples_flat)

        for name, param in self.potential.named_parameters():
            if not param.requires_grad:
                continue

            # Compute dU/dθ via autograd
            # This gives the "average" gradient, but we need per-sample

            # Strategy: Compute ⟨O * dU/dθ⟩ using autograd's chain rule
            # d/dθ ⟨O * U⟩ = ⟨O * dU/dθ⟩ (when O doesn't depend on θ)

            # Weighted sum: sum(O * U)
            weighted_U = (O_centered * U).sum()

            # Gradient of weighted sum
            grad_weighted = torch.autograd.grad(
                weighted_U,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]

            # Also compute ⟨dU/dθ⟩ for the baseline correction
            grad_mean = torch.autograd.grad(
                U.mean(),
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]

            if grad_weighted is not None and grad_mean is not None:
                # REINFORCE gradient: -β * [⟨O * dU/dθ⟩/n - ⟨O⟩ * ⟨dU/dθ⟩]
                # The O_centered already handles the baseline
                grad_reinforce = -self.beta * (grad_weighted / n_samples)
                grads[name] = grad_reinforce

        return grads

    def accumulate(
        self,
        samples: torch.Tensor,
        observable: Optional[Observable] = None,
    ):
        """Accumulate samples for batched gradient estimation.

        Call this multiple times with batches of samples, then call
        get_gradient() to compute the final gradient estimate.

        This is useful for:
        - Memory-efficient gradient estimation with large trajectories
        - Online/streaming gradient estimation

        Args:
            samples: Batch of equilibrium samples (batch_size, ..., dim)
            observable: Observable function O(x)
        """
        samples_flat = samples.reshape(-1, samples.shape[-1])
        n = samples_flat.shape[0]

        # Compute observable
        if observable is None:
            O = self.potential.energy(samples_flat)
        else:
            O = observable(samples_flat)

        if O.dim() == 0:
            O = O.expand(n)

        # Update accumulators
        self._n_samples += n

        if self._sum_O is None:
            self._sum_O = O.sum()
        else:
            self._sum_O = self._sum_O + O.sum()

        # Compute and accumulate O * dU/dθ terms
        U = self.potential.energy(samples_flat)

        for name, param in self.potential.named_parameters():
            if not param.requires_grad:
                continue

            # ⟨O * dU/dθ⟩ contribution
            weighted_U = (O * U).sum()
            grad = torch.autograd.grad(
                weighted_U, param,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]

            if grad is not None:
                if name not in self._sum_O_dU:
                    self._sum_O_dU[name] = grad.clone()
                else:
                    self._sum_O_dU[name] = self._sum_O_dU[name] + grad

            # ⟨dU/dθ⟩ contribution
            grad_U = torch.autograd.grad(
                U.sum(), param,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]

            if grad_U is not None:
                if name not in self._sum_dU:
                    self._sum_dU[name] = grad_U.clone()
                else:
                    self._sum_dU[name] = self._sum_dU[name] + grad_U

    def get_gradient(self) -> dict[str, torch.Tensor]:
        """Compute gradient from accumulated samples.

        Returns:
            Dictionary mapping parameter names to gradient estimates.

        Note:
            Resets accumulators after computing gradient.
        """
        if self._n_samples == 0:
            raise RuntimeError("No samples accumulated. Call accumulate() first.")

        n = self._n_samples
        O_mean = self._sum_O / n

        grads = {}
        for name in self._sum_O_dU:
            # REINFORCE: -β * [⟨O * dU/dθ⟩ - ⟨O⟩ * ⟨dU/dθ⟩]
            mean_O_dU = self._sum_O_dU[name] / n
            mean_dU = self._sum_dU[name] / n
            grads[name] = -self.beta * (mean_O_dU - O_mean * mean_dU)

        self._reset_accumulators()
        return grads

    def compute_variance(
        self,
        samples: torch.Tensor,
        observable: Optional[Observable] = None,
        n_bootstrap: int = 100,
    ) -> dict[str, torch.Tensor]:
        """Estimate variance of gradient estimates via bootstrap.

        Args:
            samples: Equilibrium samples (n_samples, ..., dim)
            observable: Observable function O(x)
            n_bootstrap: Number of bootstrap resamples

        Returns:
            Dictionary mapping parameter names to variance estimates.
        """
        samples_flat = samples.reshape(-1, samples.shape[-1])
        n_samples = samples_flat.shape[0]

        # Collect bootstrap gradient estimates
        grad_samples = {name: [] for name, p in self.potential.named_parameters()
                       if p.requires_grad}

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = torch.randint(0, n_samples, (n_samples,), device=samples.device)
            bootstrap_samples = samples_flat[idx]

            # Compute gradient on bootstrap sample
            grads = self.estimate_gradient(bootstrap_samples, observable)

            for name, grad in grads.items():
                grad_samples[name].append(grad.detach())

        # Compute variance
        variances = {}
        for name, grad_list in grad_samples.items():
            stacked = torch.stack(grad_list)
            variances[name] = stacked.var(dim=0)

        return variances


def reinforce_gradient(
    samples: torch.Tensor,
    potential: nn.Module,
    observable: Optional[Observable] = None,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Functional API for REINFORCE gradient estimation.

    Computes:
        ∇_θ ⟨O⟩ = -β [⟨O ∇_θU⟩ - ⟨O⟩⟨∇_θU⟩]

    Args:
        samples: Equilibrium samples (n_samples, ..., dim)
        potential: Potential energy function with parameters
        observable: Observable function O(x). Default: mean energy.
        beta: Inverse temperature 1/kT.

    Returns:
        Dictionary mapping parameter names to gradient estimates.

    Example:
        >>> grads = reinforce_gradient(trajectory[1000:], potential,
        ...                            observable=lambda x: x.mean())
        >>> for name, param in potential.named_parameters():
        ...     param.grad = grads.get(name, None)
        >>> optimizer.step()
    """
    estimator = ReinforceEstimator(potential, beta=beta)
    return estimator.estimate_gradient(samples, observable)


class GirsanovEstimator(nn.Module):
    """Girsanov-based path reweighting for gradient estimation.

    For Langevin dynamics dx = F_θ(x)dt + σdW, the log path probability is:

        log p(τ|θ) ∝ ∫₀ᵀ (1/σ²) F_θ(x_t) · (dx_t - F_θ(x_t)dt)

    This allows reweighting trajectories when parameters change:
        ⟨O⟩_θ' = ⟨O · w(τ)⟩_θ / ⟨w(τ)⟩_θ

    where w(τ) = exp(log p(τ|θ') - log p(τ|θ)).

    Warning: Girsanov reweighting has exponentially growing variance for long
    trajectories. For equilibrium properties, use ReinforceEstimator instead.

    Args:
        potential: Potential energy function with parameters.
        sigma: Noise scale (diffusion coefficient).
        beta: Inverse temperature.
    """

    def __init__(
        self,
        potential: nn.Module,
        sigma: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.potential = potential
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)

    def compute_log_path_score(
        self,
        trajectory: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute score ∇_θ log p(τ|θ) for a trajectory.

        Uses the Girsanov formula:
            ∇_θ log p(τ) = (1/σ²) ∫ ∇_θF(x_t) · (dx_t - F(x_t)dt)

        Args:
            trajectory: Positions over time (n_steps, ..., dim)
            dt: Time step used in simulation

        Returns:
            Log path score for gradient computation
        """
        n_steps = trajectory.shape[0]

        # Compute increments dx_t = x_{t+1} - x_t
        dx = trajectory[1:] - trajectory[:-1]  # (n_steps-1, ..., dim)

        # Compute forces at each step
        forces = torch.stack([
            self.potential.force(trajectory[t])
            for t in range(n_steps - 1)
        ])  # (n_steps-1, ..., dim)

        # Noise increment: dx - F*dt (should be ~ σ*dW)
        noise_increment = dx - forces * dt

        # Girsanov weight integrand: (1/σ²) F · dW
        integrand = (1 / self.sigma**2) * (forces * noise_increment).sum(dim=-1)

        # Integrate over trajectory
        log_score = integrand.sum(dim=0)  # Sum over time

        return log_score

    def estimate_gradient(
        self,
        trajectories: torch.Tensor,
        observable: Observable,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Estimate gradient using Girsanov reweighting.

        Uses the score function estimator in path space:
            ∇_θ ⟨O⟩ = ⟨O · ∇_θ log p(τ|θ)⟩

        Args:
            trajectories: Multiple trajectories (n_traj, n_steps, dim) or
                         single trajectory (n_steps, ..., dim)
            observable: Observable computed on trajectories
            dt: Time step used in simulation

        Returns:
            Dictionary of gradient estimates per parameter.
        """
        # Ensure batch dimension
        if trajectories.dim() == 2:
            trajectories = trajectories.unsqueeze(0)

        n_traj = trajectories.shape[0]

        # Compute observables
        O = observable(trajectories)  # (n_traj,)

        # Compute log path scores and their gradients
        grads = {}

        for i in range(n_traj):
            traj = trajectories[i]
            log_score = self.compute_log_path_score(traj, dt)

            # REINFORCE in path space: O * ∇_θ log p(τ)
            weighted_log_score = O[i] * log_score

            for name, param in self.potential.named_parameters():
                if not param.requires_grad:
                    continue

                grad = torch.autograd.grad(
                    weighted_log_score.sum(), param,
                    create_graph=True, retain_graph=True, allow_unused=True
                )[0]

                if grad is not None:
                    if name not in grads:
                        grads[name] = grad / n_traj
                    else:
                        grads[name] = grads[name] + grad / n_traj

        return grads


class ReweightingLoss(nn.Module):
    """Loss function using REINFORCE gradient for equilibrium observables.

    This creates a differentiable loss that, when backpropagated, produces
    REINFORCE gradients. This integrates with standard PyTorch optimizers.

    The trick is to create a "surrogate" loss:
        L_surrogate = -β * (O - baseline) * U

    whose gradient equals the REINFORCE gradient:
        ∇_θ L_surrogate = -β * (O - baseline) * ∇_θU = ∇_θ⟨O⟩_REINFORCE

    Example:
        >>> loss_fn = ReweightingLoss(potential, beta=1.0)
        >>>
        >>> # In training loop:
        >>> samples = integrator.run(x0, force_fn, ...)[burn_in:]
        >>> loss = loss_fn(samples, observable=lambda x: x.mean())
        >>> loss.backward()
        >>> optimizer.step()

    Args:
        potential: Potential energy function with parameters.
        beta: Inverse temperature.
    """

    def __init__(self, potential: nn.Module, beta: float = 1.0):
        super().__init__()
        self.potential = potential
        self.beta = beta

    def forward(
        self,
        samples: torch.Tensor,
        observable: Optional[Observable] = None,
        detach_baseline: bool = True,
    ) -> torch.Tensor:
        """Compute surrogate loss for REINFORCE gradient.

        Args:
            samples: Equilibrium samples (n_samples, ..., dim)
            observable: Observable O(x). Default: negative energy (minimizes energy).
            detach_baseline: Whether to detach baseline from gradient computation.

        Returns:
            Scalar loss whose gradient is the REINFORCE gradient.
        """
        samples_flat = samples.reshape(-1, samples.shape[-1])
        n_samples = samples_flat.shape[0]

        # Compute observable
        if observable is None:
            O = -self.potential.energy(samples_flat)  # Minimize energy
        else:
            O = observable(samples_flat)

        if O.dim() == 0:
            O = O.expand(n_samples)

        # Baseline: mean observable (detached to not affect gradient)
        if detach_baseline:
            baseline = O.mean().detach()
        else:
            baseline = O.mean()

        # Compute energies
        U = self.potential.energy(samples_flat)

        # Surrogate loss: -β * (O - baseline) * U
        # Gradient: -β * (O - baseline) * ∇_θU
        # This equals the REINFORCE gradient
        surrogate = -self.beta * ((O - baseline) * U).mean()

        # Also add the actual loss value for monitoring
        # (gradients only come from surrogate term)
        with torch.no_grad():
            actual_loss = O.mean()

        return surrogate + 0 * actual_loss  # Keep graph but value is surrogate

    def loss_and_observable(
        self,
        samples: torch.Tensor,
        observable: Optional[Observable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both surrogate loss and observable value.

        Returns:
            Tuple of (loss for backprop, observable mean for monitoring)
        """
        samples_flat = samples.reshape(-1, samples.shape[-1])
        n_samples = samples_flat.shape[0]

        if observable is None:
            O = -self.potential.energy(samples_flat)
        else:
            O = observable(samples_flat)

        if O.dim() == 0:
            O = O.expand(n_samples)

        baseline = O.mean().detach()
        U = self.potential.energy(samples_flat)

        surrogate = -self.beta * ((O - baseline) * U).mean()
        observable_value = O.mean()

        return surrogate, observable_value


if __name__ == "__main__":
    """Demo: REINFORCE gradient estimation for equilibrium observables.

    This demo validates the REINFORCE implementation by:
    1. Comparing REINFORCE vs BPTT gradients on Double-Well potential
    2. Showing gradient variance reduction with sample size (Double-Well)
    3. Demonstrating parameter optimization: learn k to match target ⟨x²⟩
    4. 2D Muller-Brown potential gradient estimation
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from .potentials import Harmonic, DoubleWell, DoubleWell2D, MullerBrown
    from .integrators import OverdampedLangevin, BAOAB

    # Plotting style (matching other modules)
    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10.0,
        "axes.labelpad": 5.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.framealpha": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "lines.linewidth": 1.5,
    })

    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print("=" * 60)
    print("REINFORCE Gradient Estimator Demo")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Colors
    colors = {
        'reinforce': '#1f77b4',  # Blue
        'bptt': '#ff7f0e',       # Orange
        'theory': '#2ca02c',     # Green
        'variance': '#d62728',   # Red
        'target': '#9467bd',     # Purple
    }

    # =========================================================================
    # Panel 1: Harmonic - REINFORCE vs BPTT vs Theory
    # =========================================================================
    ax = axes[0, 0]
    print("\n[1] Harmonic: REINFORCE vs BPTT vs Theory...")

    # For harmonic potential U = k*x²/2:
    # ⟨x²⟩ = kT/k (exact)
    # d⟨x²⟩/dk = -kT/k² (exact)

    kT = 1.0
    k_values = np.linspace(0.5, 3.0, 8)

    reinforce_grads_h = []
    bptt_grads_h = []
    theory_grads_h = []

    for k_val in k_values:
        torch.manual_seed(42)

        # Theory
        theory_grad = -kT / (k_val ** 2)
        theory_grads_h.append(theory_grad)

        # BPTT: Run dynamics and backprop through trajectory
        potential_bptt = Harmonic(k=k_val)
        integrator = OverdampedLangevin(gamma=1.0, kT=kT)
        x0 = torch.randn(100, 1)
        traj = integrator.run(x0, potential_bptt.force, dt=0.01, n_steps=1000, store_every=5)
        samples_bptt = traj[50:].reshape(-1, 1)  # After burn-in
        obs_bptt = (samples_bptt ** 2).mean()
        obs_bptt.backward()
        bptt_grads_h.append(potential_bptt.k.grad.item())

        # REINFORCE: Use same samples but detached
        potential_rf = Harmonic(k=k_val)
        estimator = ReinforceEstimator(potential_rf, beta=1.0/kT)
        observable = lambda x: (x ** 2).sum(dim=-1)
        grads = estimator.estimate_gradient(samples_bptt.detach(), observable=observable)
        reinforce_grads_h.append(grads['k'].item())

    ax.plot(k_values, theory_grads_h, 'o-', color=colors['theory'], label='Theory', lw=2, ms=7)
    ax.plot(k_values, bptt_grads_h, 's--', color=colors['bptt'], label='BPTT', lw=2, ms=6, alpha=0.8)
    ax.plot(k_values, reinforce_grads_h, '^:', color=colors['reinforce'], label='REINFORCE', lw=2, ms=6)
    ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Spring constant k')
    ax.set_ylabel('d⟨x²⟩/dk')
    ax.set_title('Harmonic: Gradient Methods Comparison', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_axisbelow(True)

    print(f"   At k=1.0: Theory={theory_grads_h[2]:.4f}, BPTT={bptt_grads_h[2]:.4f}, REINFORCE={reinforce_grads_h[2]:.4f}")

    # =========================================================================
    # Panel 2: Variance reduction on Double-Well
    # =========================================================================
    ax = axes[0, 1]
    print("\n[2] Variance reduction (Double-Well)...")

    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
    variances = []
    means = []

    torch.manual_seed(123)
    # Generate one large trajectory and subsample
    potential_var = DoubleWell(barrier_height=1.0)
    integrator_var = BAOAB(gamma=1.0, kT=0.5, mass=1.0)
    x0_var = torch.randn(200, 1)
    traj_var, _ = integrator_var.run(x0_var, None, potential_var.force, dt=0.01, n_steps=10000, store_every=5)
    all_samples = traj_var[200:].reshape(-1, 1).detach()

    for n_samples in sample_sizes:
        estimator = ReinforceEstimator(potential_var, beta=2.0)
        observable = lambda x: (x ** 2).sum(dim=-1)

        # Bootstrap variance estimation
        n_bootstrap = 50
        grad_estimates = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, len(all_samples), (n_samples,))
            bootstrap_samples = all_samples[idx]
            grads = estimator.estimate_gradient(bootstrap_samples, observable=observable)
            grad_estimates.append(grads['barrier_height'].item())

        variances.append(np.var(grad_estimates))
        means.append(np.mean(grad_estimates))

    ax.loglog(sample_sizes, variances, 'o-', color=colors['variance'], lw=2, ms=8, label='Empirical variance')

    # Theoretical 1/N scaling
    ref_var = variances[0] * sample_sizes[0]
    theoretical_var = [ref_var / n for n in sample_sizes]
    ax.loglog(sample_sizes, theoretical_var, '--', color='gray', lw=1.5, label='1/N scaling', alpha=0.7)

    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Gradient variance')
    ax.set_title('Variance Reduction (Double-Well)', fontweight='bold')
    ax.legend()
    ax.set_axisbelow(True)

    print(f"   Variance at N=100: {variances[0]:.6f}")
    print(f"   Variance at N=10000: {variances[-1]:.6f}")
    print(f"   Reduction factor: {variances[0]/variances[-1]:.1f}x")

    # =========================================================================
    # Panel 3: Optimize spring constant to match target ⟨x²⟩ - BPTT vs REINFORCE
    # =========================================================================
    ax = axes[1, 0]
    print("\n[3] Optimizing k to match target ⟨x²⟩ (BPTT vs REINFORCE)...")

    # Goal: Find k such that ⟨x²⟩ = target_obs
    # For harmonic at temperature T: ⟨x²⟩ = T/k
    # So if target=0.5 and T=1.0, optimal k=2.0

    kT_opt = 1.0
    target_obs = 0.5  # Target ⟨x²⟩
    k_init = 0.5      # Start with k=0.5 (⟨x²⟩ = 2.0, too high)
    k_optimal = kT_opt / target_obs  # = 2.0
    n_epochs = 150
    lr_init = 0.1

    # --- REINFORCE optimization ---
    torch.manual_seed(42)
    potential_rf = Harmonic(k=k_init)
    k_history_rf = [k_init]
    obs_history_rf = []

    for epoch in range(n_epochs):
        current_k = potential_rf.k.item()
        n_samples = 5000  # More samples for lower variance
        # Exact equilibrium samples for harmonic: p(x) = N(0, kT/k)
        samples = torch.randn(n_samples, 1) * np.sqrt(kT_opt / current_k)

        current_obs = (samples ** 2).mean().item()
        obs_history_rf.append(current_obs)

        # REINFORCE gradient: d⟨x²⟩/dk
        estimator = ReinforceEstimator(potential_rf, beta=1.0/kT_opt)
        observable = lambda x: (x ** 2).sum(dim=-1)
        grads = estimator.estimate_gradient(samples, observable=observable)

        # Loss: L = (⟨x²⟩ - target)², gradient: dL/dk = 2*(⟨x²⟩ - target) * d⟨x²⟩/dk
        loss_grad = 2 * (current_obs - target_obs) * grads['k'].item()

        # SGD update with learning rate decay
        lr = lr_init / (1 + epoch * 0.02)
        with torch.no_grad():
            potential_rf.k -= lr * loss_grad
            potential_rf.k.clamp_(min=0.1, max=10.0)
        k_history_rf.append(potential_rf.k.item())

    # --- BPTT optimization ---
    torch.manual_seed(42)
    potential_bptt = Harmonic(k=k_init)
    k_history_bptt = [k_init]
    obs_history_bptt = []
    integrator = OverdampedLangevin(gamma=1.0, kT=kT_opt)

    for epoch in range(n_epochs):
        # Run dynamics to get samples (with gradient tracking)
        x0 = torch.randn(200, 1)  # More walkers
        traj = integrator.run(x0, potential_bptt.force, dt=0.01, n_steps=300, store_every=3)
        samples = traj[30:].reshape(-1, 1)  # After burn-in

        current_obs = (samples ** 2).mean()
        obs_history_bptt.append(current_obs.item())

        # Loss: (⟨x²⟩ - target)²
        loss = (current_obs - target_obs) ** 2
        loss.backward()

        # SGD update with learning rate decay
        lr = lr_init / (1 + epoch * 0.02)
        with torch.no_grad():
            grad = potential_bptt.k.grad
            if grad is not None and torch.isfinite(grad):
                potential_bptt.k -= lr * grad
                potential_bptt.k.clamp_(min=0.1, max=10.0)
            potential_bptt.k.grad = None
        k_history_bptt.append(potential_bptt.k.item())

    # Plot both
    epochs = range(len(k_history_rf))
    ax.plot(epochs, k_history_rf, '-', color=colors['reinforce'], lw=2, label='REINFORCE', alpha=0.9)
    ax.plot(epochs, k_history_bptt, '--', color=colors['bptt'], lw=2, label='BPTT', alpha=0.9)
    ax.axhline(k_optimal, color=colors['theory'], ls=':', lw=2, label=f'Optimal k={k_optimal:.1f}')
    ax.set_xlabel('Optimization epoch')
    ax.set_ylabel('Spring constant k')
    ax.set_title(f'Optimize k for target ⟨x²⟩={target_obs}', fontweight='bold')
    ax.legend(loc='right', fontsize=9)
    ax.set_axisbelow(True)

    # Add inset showing ⟨x²⟩ convergence
    ax_ins2 = ax.inset_axes([0.55, 0.15, 0.4, 0.35])
    ax_ins2.plot(range(len(obs_history_rf)), obs_history_rf, '-', color=colors['reinforce'], lw=1.5, label='RF')
    ax_ins2.plot(range(len(obs_history_bptt)), obs_history_bptt, '--', color=colors['bptt'], lw=1.5, label='BPTT')
    ax_ins2.axhline(target_obs, color=colors['target'], ls=':', lw=1.5)
    ax_ins2.set_xlabel('Epoch', fontsize=8)
    ax_ins2.set_ylabel('⟨x²⟩', fontsize=8)
    ax_ins2.tick_params(labelsize=7)
    ax_ins2.legend(fontsize=7, loc='upper right')
    ax_ins2.set_title('Observable', fontsize=9)

    print(f"   Initial k: {k_init:.2f} (⟨x²⟩ = {kT_opt/k_init:.2f})")
    print(f"   Optimal k: {k_optimal:.2f} (⟨x²⟩ = {target_obs:.2f})")
    print(f"   REINFORCE final k: {k_history_rf[-1]:.3f} (⟨x²⟩ ≈ {obs_history_rf[-1]:.3f})")
    print(f"   BPTT final k: {k_history_bptt[-1]:.3f} (⟨x²⟩ ≈ {obs_history_bptt[-1]:.3f})")

    # =========================================================================
    # Panel 4: REINFORCE vs BPTT - Stability with trajectory length
    # =========================================================================
    ax = axes[1, 1]
    print("\n[4] REINFORCE vs BPTT: Gradient stability...")

    # This is the KEY experiment showing REINFORCE advantage:
    # BPTT gradients can explode with long trajectories due to chaos
    # REINFORCE remains stable

    kT_stab = 1.0
    k_true = 1.0
    theory_grad = -kT_stab / (k_true ** 2)  # = -1.0

    trajectory_lengths = [100, 200, 500, 1000, 2000, 5000]
    n_trials = 10

    bptt_means = []
    bptt_stds = []
    reinforce_means = []
    reinforce_stds = []

    for n_steps in trajectory_lengths:
        bptt_trials = []
        rf_trials = []

        for trial in range(n_trials):
            torch.manual_seed(trial * 100 + n_steps)

            # BPTT
            potential_bptt = Harmonic(k=k_true)
            integrator = OverdampedLangevin(gamma=1.0, kT=kT_stab)
            x0 = torch.randn(50, 1)
            traj = integrator.run(x0, potential_bptt.force, dt=0.01, n_steps=n_steps, store_every=1)
            samples = traj[n_steps//4:].reshape(-1, 1)

            obs = (samples ** 2).mean()
            obs.backward()
            if potential_bptt.k.grad is not None and torch.isfinite(potential_bptt.k.grad):
                bptt_trials.append(potential_bptt.k.grad.item())

            # REINFORCE
            potential_rf = Harmonic(k=k_true)
            estimator = ReinforceEstimator(potential_rf, beta=1.0/kT_stab)
            observable = lambda x: (x ** 2).sum(dim=-1)
            grads = estimator.estimate_gradient(samples.detach(), observable=observable)
            rf_trials.append(grads['k'].item())

        bptt_means.append(np.mean(bptt_trials) if bptt_trials else np.nan)
        bptt_stds.append(np.std(bptt_trials) if bptt_trials else np.nan)
        reinforce_means.append(np.mean(rf_trials))
        reinforce_stds.append(np.std(rf_trials))

    # Plot with error bars
    ax.errorbar(trajectory_lengths, bptt_means, yerr=bptt_stds, fmt='s-',
                color=colors['bptt'], label='BPTT', lw=2, ms=6, capsize=3, alpha=0.8)
    ax.errorbar(trajectory_lengths, reinforce_means, yerr=reinforce_stds, fmt='^-',
                color=colors['reinforce'], label='REINFORCE', lw=2, ms=6, capsize=3)
    ax.axhline(theory_grad, color=colors['theory'], ls='--', lw=2, label=f'Theory: {theory_grad:.1f}')
    ax.set_xlabel('Trajectory length (steps)')
    ax.set_ylabel('Gradient estimate')
    ax.set_title('Gradient Stability vs Trajectory Length', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_axisbelow(True)
    ax.set_xscale('log')

    print(f"   Theory: {theory_grad:.2f}")
    print(f"   BPTT at 5000 steps: {bptt_means[-1]:.4f} ± {bptt_stds[-1]:.4f}")
    print(f"   REINFORCE at 5000 steps: {reinforce_means[-1]:.4f} ± {reinforce_stds[-1]:.4f}")

    # =========================================================================
    # Save figure
    # =========================================================================
    plt.savefig(os.path.join(assets_dir, "gradient_estimators.png"), dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"\n[+] Saved plot to assets/gradient_estimators.png")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary: REINFORCE vs BPTT Comparison")
    print("=" * 60)
    print(f"{'Metric':<30} | {'BPTT':<15} | {'REINFORCE':<15}")
    print("-" * 60)
    print(f"{'Memory scaling':<30} | {'O(T)':<15} | {'O(1)':<15}")
    print(f"{'Chaos sensitivity':<30} | {'High':<15} | {'None':<15}")
    print(f"{'Variance':<30} | {'Low':<15} | {'Medium-High':<15}")
    print(f"{'Requires differentiable sim':<30} | {'Yes':<15} | {'No':<15}")
    print(f"{'Works at equilibrium':<30} | {'Yes':<15} | {'Yes':<15}")
    print(f"{'Works for transients':<30} | {'Yes':<15} | {'No':<15}")
    print("=" * 60)
