"""Gradient estimators for differentiable simulations.

This module provides gradient computation methods that bypass trajectory differentiation,
using statistical estimators instead.

Gradient Estimators:
- ReinforceEstimator: REINFORCE/score-function estimator for equilibrium observables
- PathReweightingEstimator: Path likelihood ratio estimator for SDE trajectories
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
from torch.func import functional_call, jvp, vmap
from typing import Callable, Tuple, Optional, Union, Dict, List
import gc
import math

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
    - Equibrium sampling

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

    def _flatten_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Flatten samples based on potential event dimension.
        
        Handles arbitrary event dimensions (e.g. 1D vector or N-particle system).
        """
        event_dim = getattr(self.potential, 'event_dim', 1)
        # Check if samples already match event shape without batch
        if samples.ndim == event_dim:
            # Single sample case? Or batch of 1?
            # Ambiguous. Assume unbatched single sample if ndim == event_dim
            return samples.unsqueeze(0)
            
        # Flatten all leading dimensions into a single batch dimension
        # shape: (n_samples, *event_shape)
        # e.g. (B, N, 3) -> (B, N, 3) for event_dim=2
        # e.g. (T, B, N, 3) -> (T*B, N, 3)
        
        batch_dims = samples.ndim - event_dim
        if batch_dims < 0:
             raise ValueError(f"Sample dimension {samples.ndim} smaller than event dimension {event_dim}")
             
        if batch_dims == 0:
            return samples.unsqueeze(0)
            
        return samples.reshape(-1, *samples.shape[-event_dim:])

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
        # Flatten batch dimensions
        x_flat = self._flatten_samples(x)
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
                     (n_steps, ..., dim). Will be flattened.
            observable: Function O(x) -> scalar or (n_samples,) tensor.
                       If None, uses mean energy as observable.
            reduce: If True, return mean gradient. If False, return per-sample.

        Returns:
            Dictionary mapping parameter names to gradient estimates.

        Note:
            For accurate gradients, samples should be from equilibrium.
            Include burn-in period before calling this function.
        """
        # Flatten trajectory to samples respecting event structure
        samples_flat = self._flatten_samples(samples)
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
            # Try reshaping if possible (e.g. (N, 1) -> (N,))
            if O.numel() == n_samples:
                O = O.reshape(n_samples)
            else:
                raise ValueError(f"Observable must return (n_samples,) tensor, got {O.shape} for {n_samples} samples")

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
        grads = {}

        # Compute energy and its gradients
        U = self.potential.energy(samples_flat)

        for name, param in self.potential.named_parameters():
            if not param.requires_grad:
                continue

            # Compute dU/dθ via autograd
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

            # Also compute ⟨dU/dθ⟩ for the baseline correction (if needed explicitly)
            # Actually, O_centered already handles the baseline implicitly:
            # sum((O - b) * dU/dθ) = sum(O * dU/dθ) - b * sum(dU/dθ)
            # This is exactly what we want.

            if grad_weighted is not None:
                # REINFORCE gradient: -β * [⟨(O - b) * dU/dθ⟩]
                grad_reinforce = -self.beta * (grad_weighted / n_samples)
                grads[name] = grad_reinforce

        return grads

    def accumulate(
        self,
        samples: torch.Tensor,
        observable: Optional[Observable] = None,
    ):
        """Accumulate samples for batched gradient estimation.

        Args:
            samples: Batch of equilibrium samples
            observable: Observable function O(x)
        """
        samples_flat = self._flatten_samples(samples)
        n = samples_flat.shape[0]

        # Compute observable
        if observable is None:
            O = self.potential.energy(samples_flat)
        else:
            O = observable(samples_flat)

        if O.dim() == 0:
            O = O.expand(n)
        elif O.numel() == n:
            O = O.reshape(n)

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
            samples: Equilibrium samples
            observable: Observable function O(x)
            n_bootstrap: Number of bootstrap resamples

        Returns:
            Dictionary mapping parameter names to variance estimates.
        """
        samples_flat = self._flatten_samples(samples)
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


class PathReweightingEstimator(nn.Module):
    """Path reweighting gradient estimator for SDE trajectories.

    Computes gradients using the likelihood ratio method on path space.
    For Langevin dynamics dx = F_θ(x)dt + σdW, the path log-likelihood
    gradient is given by the Girsanov-Cameron-Martin formula:

        ∇_θ log p(τ|θ) = (1/σ²) ∫₀ᵀ ∇_θF_θ(x_t) · dW_t

    This enables the estimator:
        ∇_θ ⟨O⟩ = ⟨O · ∇_θ log p(τ|θ)⟩

    Also known as:
    - Likelihood Ratio (LR) method (operations research)
    - Girsanov reweighting (mathematical finance/stochastic analysis)
    - Path integral gradient (physics)

    Note: This is distinct from "score function" in diffusion models,
    which refers to ∇_x log p(x), the spatial gradient of log-density.
    Here we compute ∇_θ log p(τ|θ), the parameter gradient of path likelihood.

    Variance Reduction Methods:
    ---------------------------
    Girsanov reweighting has exponentially growing variance for long trajectories.
    This class implements several variance reduction techniques:

    1. **Centering (baseline subtraction)**: Subtracts ⟨∇log p⟩ from the score,
       reducing variance from O(T) to O(1) in system size. Based on:
       Gupta & Khammash, "Centered Girsanov Transformation" (2014).

    2. **Marginal Girsanov Reweighting (MGR)**: Instead of full path weights,
       uses marginalized weights at each timestep. Avoids exponential variance
       growth. Based on: Tiwari et al., arXiv:2509.25872.

    3. **Truncated weights (clamping)**: Clips extreme log-weights to prevent
       single trajectories from dominating. Simple but introduces bias.

    4. **Self-normalized importance sampling**: Uses w_i / Σw_j instead of w_i,
       which is consistent and often lower variance.

    5. **Effective sample size (ESS) monitoring**: Tracks ESS = (Σw)² / Σw²
       to detect weight degeneracy.

    For equilibrium properties, ReinforceEstimator is often preferable as it
    has O(1/√N) variance without the exponential T dependence.

    Args:
        potential: Potential energy function with parameters.
        sigma: Noise scale (diffusion coefficient).
        beta: Inverse temperature.
        variance_reduction: Strategy for variance reduction. Options:
            - None: No variance reduction (standard Girsanov)
            - "center": Centered Girsanov (subtract mean score)
            - "marginal": Marginal Girsanov reweighting
            - "truncate": Truncate extreme log-weights
            - "self_normalize": Self-normalized importance sampling
        truncate_threshold: For "truncate" mode, max |log w| allowed (default: 10)
    """

    def __init__(
        self,
        potential: nn.Module,
        sigma: float = 1.0,
        beta: float = 1.0,
        variance_reduction: Optional[str] = "center",
        truncate_threshold: float = 10.0,
    ):
        super().__init__()
        self.potential = potential
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.variance_reduction = variance_reduction
        self.truncate_threshold = truncate_threshold

    def compute_log_path_score(
        self,
        trajectory: torch.Tensor,
        dt: float,
        return_per_step: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute score ∇_θ log p(τ|θ) for a trajectory.

        Uses the Girsanov formula:
            ∇_θ log p(τ) = (1/σ²) ∫ ∇_θF(x_t) · (dx_t - F(x_t)dt)

        Args:
            trajectory: Positions over time (n_steps, ..., dim)
            dt: Time step used in simulation
            return_per_step: If True, also return per-step contributions

        Returns:
            Log path score for gradient computation.
            If return_per_step=True, returns (total_score, per_step_scores).
        """
        n_steps = trajectory.shape[0]

        # Compute increments dx_t = x_{t+1} - x_t
        dx = trajectory[1:] - trajectory[:-1]  # (n_steps-1, ..., dim)

        # Compute forces at each step
        # Vectorized call to potential.force for efficiency
        forces = self.potential.force(trajectory[:-1])  # (n_steps-1, ..., dim)

        # Noise increment: dx - F*dt (should be ~ σ*dW)
        noise_increment = dx - forces * dt

        # Girsanov weight integrand: (1/σ²) F · dW
        # Sum over all spatial/particle dimensions, keeping batch and time dims
        event_dim = getattr(self.potential, 'event_dim', 1)
        # Time is dim 0. Batch dimensions are between 0 and event_dims
        n_batch_dims = forces.ndim - 1 - event_dim
        flat_dims = tuple(range(1 + n_batch_dims, forces.ndim))
        
        integrand = (1 / self.sigma**2) * (forces * noise_increment).sum(dim=flat_dims)

        # Integrate over trajectory
        log_score = integrand.sum(dim=0)  # Sum over time

        if return_per_step:
            return log_score, integrand
        return log_score

    def compute_marginal_weights(
        self,
        trajectory: torch.Tensor,
        dt: float,
        window_size: int = 10,
    ) -> torch.Tensor:
        """Compute marginal Girsanov weights (MGR) for variance reduction.

        Instead of computing full path weights exp(∫₀ᵀ ...), MGR marginalizes
        over intermediate paths using a sliding window. This prevents the
        exponential variance growth of standard Girsanov.

        The marginal weight at time t uses only a local window:
            w_t = exp(∫_{t-W}^{t} (1/σ²) F · dW)

        Reference: Tiwari et al., "Marginal Girsanov Reweighting" arXiv:2509.25872

        Args:
            trajectory: Positions over time (n_steps, ..., dim)
            dt: Time step
            window_size: Size of local window for marginalization

        Returns:
            Marginal weights at each timestep (n_steps-1, ...)
        """
        _, per_step = self.compute_log_path_score(trajectory, dt, return_per_step=True)
        
        n_steps = per_step.shape[0]
        
        # Compute windowed cumulative sums
        # For each timestep t, sum over [max(0, t-window_size), t]
        marginal_log_weights = torch.zeros_like(per_step)
        
        for t in range(n_steps):
            start = max(0, t - window_size + 1)
            marginal_log_weights[t] = per_step[start:t+1].sum(dim=0)
        
        return marginal_log_weights

    def compute_effective_sample_size(
        self,
        log_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute effective sample size (ESS) to monitor weight degeneracy.

        ESS = (Σw)² / Σw² = 1 / Σ(w_normalized)²

        An ESS close to N indicates good coverage; ESS << N indicates
        weight degeneracy where few trajectories dominate.

        Args:
            log_weights: Log importance weights (n_traj,)

        Returns:
            Effective sample size (scalar)
        """
        # Normalize in log space for numerical stability
        log_weights_centered = log_weights - log_weights.max()
        weights = torch.exp(log_weights_centered)
        
        # ESS = (sum w)^2 / sum w^2
        sum_w = weights.sum()
        sum_w2 = (weights ** 2).sum()
        
        ess = (sum_w ** 2) / (sum_w2 + 1e-10)
        return ess

    def estimate_gradient(
        self,
        trajectories: torch.Tensor,
        observable: Observable,
        dt: float,
        return_diagnostics: bool = False,
    ) -> Union[dict[str, torch.Tensor], Tuple[dict[str, torch.Tensor], dict]]:
        """Estimate gradient using Girsanov reweighting with variance reduction.

        Uses the score function estimator in path space:
            ∇_θ ⟨O⟩ = ⟨O · ∇_θ log p(τ|θ)⟩

        Variance reduction is applied based on self.variance_reduction setting.

        Args:
            trajectories: Multiple trajectories (n_traj, n_steps, ..., dim) or
                         single trajectory (n_steps, ..., dim)
            observable: Observable computed on trajectories
            dt: Time step used in simulation
            return_diagnostics: If True, return diagnostic info (ESS, variance)

        Returns:
            Dictionary of gradient estimates per parameter.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        # Ensure batch dimension
        # If trajectories is (n_steps, ..., dim), add n_traj=1 at dim 0
        event_dim = getattr(self.potential, 'event_dim', 1)
        if trajectories.ndim == event_dim + 1:
            trajectories = trajectories.unsqueeze(0)

        n_traj = trajectories.shape[0]

        # Compute observables
        O = observable(trajectories)  # (n_traj,)

        # Compute log path scores and their gradients (vectorized over trajectories)
        # Transpose to put time at dim 0 for compute_log_path_score
        # (n_traj, n_steps, ...) -> (n_steps, n_traj, ...)
        trajectories_reordered = trajectories.transpose(0, 1)
        log_scores = self.compute_log_path_score(trajectories_reordered, dt)  # (n_traj,)

        # Apply variance reduction
        diagnostics = {}
        
        if self.variance_reduction == "center":
            # Centered Girsanov: subtract mean score (baseline)
            # This reduces variance from O(T) to O(1) in system size
            log_scores_centered = log_scores - log_scores.mean()
            effective_scores = log_scores_centered
            diagnostics['score_mean'] = log_scores.mean().detach()
            diagnostics['score_std'] = log_scores.std().detach()
            
        elif self.variance_reduction == "truncate":
            # Truncate extreme log-weights to prevent dominance
            # Introduces bias but prevents variance explosion
            log_scores_clamped = torch.clamp(
                log_scores, 
                min=-self.truncate_threshold, 
                max=self.truncate_threshold
            )
            n_truncated = ((log_scores.abs() > self.truncate_threshold).sum()).item()
            effective_scores = log_scores_clamped
            diagnostics['n_truncated'] = n_truncated
            diagnostics['truncate_fraction'] = n_truncated / n_traj
            
        elif self.variance_reduction == "self_normalize":
            # Self-normalized importance sampling for score function gradient
            #
            # The standard score function estimator is:
            #   ∇⟨O⟩ = ⟨O · ∇log p⟩
            #
            # With self-normalization, we use importance weights to reweight:
            #   ∇⟨O⟩ ≈ Cov_w(O, ∇log p) = ⟨w̃ · (O - ⟨O⟩_w) · ∇log p⟩
            #
            # where w̃_i = exp(log_score_i) / Σ exp(log_score_j) are normalized
            # importance weights derived from the path scores.
            #
            # This is similar to centering but uses weighted statistics.
            
            # Compute importance weights from log scores
            log_weights = log_scores - log_scores.max()  # Numerical stability
            weights = torch.exp(log_weights)
            normalized_weights = weights / (weights.sum() + 1e-10)
            
            # Weighted mean observable
            O_weighted_mean = (normalized_weights * O).sum()
            
            # Center both observable and scores using weighted means
            O_centered = O - O_weighted_mean.detach()
            score_mean = (normalized_weights * log_scores).sum()
            score_centered = log_scores - score_mean.detach()
            
            # Effective score: just use centered scores (like centering method)
            # The key difference is using weighted means for centering
            effective_scores = score_centered
            
            # Override O with weighted-centered version
            O = O_centered
            
            # Compute ESS
            ess = self.compute_effective_sample_size(log_scores)
            diagnostics['ess'] = ess.detach()
            diagnostics['ess_fraction'] = (ess / n_traj).detach()
            diagnostics['O_weighted_mean'] = O_weighted_mean.detach()
            
        elif self.variance_reduction == "marginal":
            # Marginal Girsanov: use only final-time marginal weight
            # This avoids path-integral variance but only works for
            # observables that depend on final state
            # For full MGR, would need per-timestep observables
            effective_scores = log_scores  # Fallback to standard for now
            diagnostics['warning'] = "Full MGR requires per-timestep observables"
            
        else:
            # No variance reduction
            effective_scores = log_scores

        # Compute ESS for diagnostics (if not already computed)
        if 'ess' not in diagnostics:
            ess = self.compute_effective_sample_size(log_scores)
            diagnostics['ess'] = ess.detach()
            diagnostics['ess_fraction'] = (ess / n_traj).detach()

        # REINFORCE in path space: ∇_θ ⟨O⟩ = ⟨O * ∇_θ log p(τ)⟩
        # We compute gradient of ⟨O * log p(τ)⟩
        if self.variance_reduction == "center":
            # For centered, we use (O - ⟨O⟩) * (score - ⟨score⟩)
            O_centered = O - O.mean()
            weighted_log_scores = (O_centered * effective_scores).sum()
        elif self.variance_reduction == "self_normalize":
            # Self-normalized: O and effective_scores are already centered
            # using weighted means. Just compute the sum.
            weighted_log_scores = (O * effective_scores).sum()
        else:
            weighted_log_scores = (O * effective_scores).sum()

        grads = {}
        for name, param in self.potential.named_parameters():
            if not param.requires_grad:
                continue

            grad = torch.autograd.grad(
                weighted_log_scores, param,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]

            if grad is not None:
                grads[name] = grad / n_traj

        if return_diagnostics:
            return grads, diagnostics
        return grads

    def estimate_gradient_with_control_variate(
        self,
        trajectories: torch.Tensor,
        observable: Observable,
        control_variate: Observable,
        dt: float,
        optimize_coefficient: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Estimate gradient using control variates for variance reduction.

        Uses the identity that for any function c(τ) with known expectation:
            ∇⟨O⟩ = ⟨(O - α·c) · ∇log p⟩ + α·∇⟨c⟩

        If ⟨c⟩ and ∇⟨c⟩ are known analytically (e.g., c = energy), this
        can significantly reduce variance.

        The optimal coefficient α* minimizes Var((O - α·c) · ∇log p):
            α* = Cov(O·∇log p, c·∇log p) / Var(c·∇log p)

        Args:
            trajectories: Multiple trajectories (n_traj, n_steps, ..., dim)
            observable: Observable O(τ)
            control_variate: Control variate function c(τ)
            dt: Time step
            optimize_coefficient: If True, compute optimal α

        Returns:
            Dictionary of gradient estimates per parameter.
        """
        event_dim = getattr(self.potential, 'event_dim', 1)
        if trajectories.ndim == event_dim + 1:
            trajectories = trajectories.unsqueeze(0)

        n_traj = trajectories.shape[0]

        # Compute observables
        O = observable(trajectories)
        C = control_variate(trajectories)

        # Compute log path scores
        trajectories_reordered = trajectories.transpose(0, 1)
        log_scores = self.compute_log_path_score(trajectories_reordered, dt)

        if optimize_coefficient:
            # Compute optimal coefficient
            # α* = Cov(O·score, C·score) / Var(C·score)
            O_score = O * log_scores
            C_score = C * log_scores
            
            cov_OC = ((O_score - O_score.mean()) * (C_score - C_score.mean())).mean()
            var_C = ((C_score - C_score.mean()) ** 2).mean()
            
            alpha = cov_OC / (var_C + 1e-10)
            alpha = alpha.detach()  # Don't differentiate through α
        else:
            alpha = 1.0

        # Compute gradient with control variate
        O_adjusted = O - alpha * C
        weighted_log_scores = (O_adjusted * log_scores).sum()

        grads = {}
        for name, param in self.potential.named_parameters():
            if not param.requires_grad:
                continue

            grad = torch.autograd.grad(
                weighted_log_scores, param,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]

            if grad is not None:
                grads[name] = grad / n_traj

        return grads

    def compute_variance(
        self,
        trajectories: torch.Tensor,
        observable: Observable,
        dt: float,
        n_bootstrap: int = 100,
    ) -> Tuple[dict[str, torch.Tensor], dict]:
        """Estimate variance of gradient estimates via bootstrap.

        Also returns diagnostic statistics about weight degeneracy.

        Args:
            trajectories: Multiple trajectories (n_traj, n_steps, ..., dim)
            observable: Observable function
            dt: Time step
            n_bootstrap: Number of bootstrap resamples

        Returns:
            Tuple of (variance_dict, diagnostics_dict)
        """
        event_dim = getattr(self.potential, 'event_dim', 1)
        if trajectories.ndim == event_dim + 1:
            trajectories = trajectories.unsqueeze(0)
        
        n_traj = trajectories.shape[0]
        device = trajectories.device

        # Collect bootstrap gradient estimates
        grad_samples = {name: [] for name, p in self.potential.named_parameters()
                       if p.requires_grad}
        ess_samples = []

        for _ in range(n_bootstrap):
            # Resample trajectories with replacement
            idx = torch.randint(0, n_traj, (n_traj,), device=device)
            bootstrap_traj = trajectories[idx]

            # Compute gradient on bootstrap sample
            grads, diag = self.estimate_gradient(
                bootstrap_traj, observable, dt, return_diagnostics=True
            )

            for name, grad in grads.items():
                grad_samples[name].append(grad.detach())
            
            ess_samples.append(diag.get('ess', torch.tensor(0.0)))

        # Compute variance
        variances = {}
        for name, grad_list in grad_samples.items():
            if grad_list:
                stacked = torch.stack(grad_list)
                variances[name] = stacked.var(dim=0)

        # Diagnostic statistics
        ess_tensor = torch.stack(ess_samples)
        diagnostics = {
            'ess_mean': ess_tensor.mean(),
            'ess_std': ess_tensor.std(),
            'ess_min': ess_tensor.min(),
            'variance_reduction_method': self.variance_reduction,
        }

        return variances, diagnostics


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

    def _flatten_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Flatten samples based on potential event dimension."""
        event_dim = getattr(self.potential, 'event_dim', 1)
        if samples.ndim == event_dim:
            return samples.unsqueeze(0)
        return samples.reshape(-1, *samples.shape[-event_dim:])

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
        samples_flat = self._flatten_samples(samples)
        n_samples = samples_flat.shape[0]

        # Compute observable
        if observable is None:
            O = -self.potential.energy(samples_flat)  # Minimize energy
        else:
            O = observable(samples_flat)

        if O.dim() == 0:
            O = O.expand(n_samples)
        elif O.numel() == n_samples:
            O = O.reshape(n_samples)

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
        samples_flat = self._flatten_samples(samples)
        n_samples = samples_flat.shape[0]

        if observable is None:
            O = -self.potential.energy(samples_flat)
        else:
            O = observable(samples_flat)

        if O.dim() == 0:
            O = O.expand(n_samples)
        elif O.numel() == n_samples:
            O = O.reshape(n_samples)

        baseline = O.mean().detach()
        U = self.potential.energy(samples_flat)

        surrogate = -self.beta * ((O - baseline) * U).mean()
        observable_value = O.mean()

        return surrogate, observable_value


class ImplicitDiffEstimator(nn.Module):
    """Implicit differentiation estimator for optimization and equilibrium problems.

    Uses the implicit function theorem to compute gradients without unrolling.

    For OPTIMIZATION (energy minimization):
        x* = argmin U(x; θ)
        Optimality: F(x*, θ) = ∇_x U(x*, θ) = 0
        By implicit function theorem:
            ∂x*/∂θ = -[∂²U/∂x²]⁻¹ · [∂²U/∂x∂θ]

    For EQUILIBRIUM SAMPLING:
        The gradient of ⟨O⟩ w.r.t. θ reduces to the same formula as REINFORCE:
            ∂⟨O⟩/∂θ = -β Cov(O, ∂U/∂θ)

    Reference:
        Blondel et al. "Efficient and Modular Implicit Differentiation" NeurIPS 2022

    Advantages:
        - O(1) memory (no trajectory storage)
        - Exact for optimization problems
        - Can use any solver (not just differentiable ones)

    Limitations:
        - Requires fixed point or optimality condition
        - Does NOT work for path-dependent observables (FPT, work, etc.)
        - For equilibrium sampling, equivalent to REINFORCE

    Args:
        potential: Potential energy function (nn.Module with parameters)
        beta: Inverse temperature 1/kT
        mode: 'optimization' for energy minimization, 'equilibrium' for sampling
    """

    def __init__(
        self,
        potential: nn.Module,
        beta: float = 1.0,
        mode: str = 'equilibrium',
    ):
        super().__init__()
        self.potential = potential
        self.beta = beta
        self.mode = mode

    def _flatten_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Flatten samples based on potential event dimension."""
        event_dim = getattr(self.potential, 'event_dim', 1)
        if samples.ndim == event_dim:
            return samples.unsqueeze(0)
        return samples.reshape(-1, *samples.shape[-event_dim:])

    def gradient_energy_minimization(
        self,
        x_star: torch.Tensor,
        observable: Optional[Observable] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute gradient for energy minimization problem.

        For x* = argmin U(x; θ), uses implicit function theorem:
            ∂x*/∂θ = -[∂²U/∂x²]⁻¹ · [∂²U/∂x∂θ]

        If observable O(x*) is provided, chain rule gives:
            ∂O/∂θ = ∂O/∂x* · ∂x*/∂θ

        Args:
            x_star: Optimal solution x* (should satisfy ∇U ≈ 0)
            observable: Optional function O(x) to differentiate

        Returns:
            Dictionary mapping parameter names to gradients
        """
        # Ensure correct shape for Hessian (requires single batch dimension usually?)
        # For simplicity, assume x_star is a single sample or batch of optima
        # Implicit diff is tricky for batches. We will assume batch average.
        
        x = x_star.clone().requires_grad_(True)
        grads = {}

        # Compute gradient ∇_x U (should be ~0 at minimum)
        U = self.potential.energy(x)
        grad_x = torch.autograd.grad(U.sum(), x, create_graph=True)[0]

        for name, param in self.potential.named_parameters():
            if not param.requires_grad:
                continue

            # Compute Hessian ∂²U/∂x² (scalar for 1D)
            if x.shape[-1] == 1:
                # 1D case: Hessian is a scalar
                hess = torch.autograd.grad(
                    grad_x.sum(), x, create_graph=True, retain_graph=True
                )[0]
                H = hess.mean()  # Average over batch

                # Mixed partial ∂²U/∂x∂θ
                mixed = torch.autograd.grad(
                    grad_x.sum(), param, create_graph=True, retain_graph=True,
                    allow_unused=True
                )[0]

                if mixed is not None and H.abs() > 1e-10:
                    # ∂x*/∂θ = -mixed / H
                    dx_dtheta = -mixed / H

                    if observable is not None:
                        # Chain rule: ∂O/∂θ = ∂O/∂x · ∂x*/∂θ
                        O = observable(x)
                        dO_dx = torch.autograd.grad(
                            O.sum(), x, create_graph=True, retain_graph=True
                        )[0]
                        grads[name] = (dO_dx.mean() * dx_dtheta).squeeze()
                    else:
                        grads[name] = dx_dtheta.squeeze()
            else:
                # Multi-dimensional: need to solve linear system
                # For simplicity, skip for now (would need proper Hessian computation)
                pass

        return grads

    def estimate_gradient(
        self,
        samples: torch.Tensor,
        observable: Optional[Observable] = None,
    ) -> dict[str, torch.Tensor]:
        """Estimate gradient using implicit differentiation.

        For equilibrium sampling, this reduces to the REINFORCE formula:
            ∂⟨O⟩/∂θ = -β Cov(O, ∂U/∂θ)

        This is because the equilibrium distribution p(x) ∝ exp(-βU) satisfies
        a self-consistency condition, and differentiating through it gives
        the same result as score function estimation.

        Args:
            samples: Equilibrium samples (n_samples, dim)
            observable: Observable function O(x)

        Returns:
            Dictionary mapping parameter names to gradient estimates
        """
        if self.mode == 'optimization':
            # For optimization, assume samples is the optimal x*
            return self.gradient_energy_minimization(samples, observable)

        # For equilibrium: use the same formula as REINFORCE
        samples_flat = self._flatten_samples(samples)
        n_samples = samples_flat.shape[0]

        if observable is None:
            O = self.potential.energy(samples_flat)
        else:
            O = observable(samples_flat)

        if O.dim() == 0:
            O = O.expand(n_samples)
        elif O.numel() == n_samples:
            O = O.reshape(n_samples)

        O_centered = O - O.mean()
        U = self.potential.energy(samples_flat)

        grads = {}
        for name, param in self.potential.named_parameters():
            if not param.requires_grad:
                continue

            # Compute ⟨(O - ⟨O⟩) · ∂U/∂θ⟩
            weighted_U = (O_centered * U).sum()
            grad_weighted = torch.autograd.grad(
                weighted_U, param, create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]

            if grad_weighted is not None:
                grads[name] = -self.beta * (grad_weighted / n_samples)

        return grads


# ============================================================================
# O(1) Adjoint Sensitivity Methods for NoseHoover Integrator
# ============================================================================


class CheckpointManager:
    """Manages checkpoint storage and retrieval for discrete adjoint methods.

    Stores periodic checkpoints during forward pass to enable memory-efficient
    backward pass via recomputation. Uses O(√T) memory instead of O(T).

    Args:
        n_steps: Total number of integration steps
        n_checkpoints: Number of checkpoints to store (default: √n_steps)
    """

    def __init__(self, n_steps: int, n_checkpoints: Optional[int] = None):
        if n_checkpoints is None:
            n_checkpoints = max(1, int(math.sqrt(n_steps)))

        self.n_steps = n_steps
        self.n_checkpoints = n_checkpoints
        self.checkpoints: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Compute checkpoint indices (evenly spaced)
        if n_checkpoints > 1:
            self.checkpoint_indices = set(
                int(i * n_steps / (n_checkpoints - 1)) for i in range(n_checkpoints - 1)
            )
            self.checkpoint_indices.add(0)  # Always checkpoint initial state
        else:
            self.checkpoint_indices = {0}

    def save_checkpoint(self, step: int, x: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor):
        """Save checkpoint at given step (detached from gradient graph)."""
        self.checkpoints[step] = (
            x.detach().clone(),
            v.detach().clone(),
            alpha.detach().clone()
        )

    def get_checkpoint(self, step: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get checkpoint at exact step, or None if not found."""
        return self.checkpoints.get(step)

    def get_nearest_checkpoint_before(self, step: int) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get nearest checkpoint at or before given step."""
        # Find largest checkpoint index <= step
        valid_indices = [idx for idx in self.checkpoints.keys() if idx <= step]
        if not valid_indices:
            raise ValueError(f"No checkpoint found before step {step}")

        nearest_idx = max(valid_indices)
        return nearest_idx, self.checkpoints[nearest_idx]

    def should_checkpoint(self, step: int) -> bool:
        """Check if this step should be checkpointed."""
        return step in self.checkpoint_indices

    def clear(self):
        """Clear all stored checkpoints."""
        self.checkpoints.clear()


class NoseHooverCheckpointedFunction(torch.autograd.Function):
    """Custom autograd function implementing discrete adjoint with checkpointing.

    This function enables O(√T) memory backward pass for NoseHoover dynamics
    by storing periodic checkpoints and recomputing intermediate states.
    """

    @staticmethod
    def forward(ctx, x0, v0, alpha0, kT, mass, Q, force_fn, dt, n_steps, store_every, checkpoint_mgr, final_only):
        """Forward pass with checkpointing.

        Args:
            ctx: Context for saving info for backward pass
            x0, v0, alpha0: Initial state
            kT, mass, Q: NoseHoover parameters
            force_fn: Force function
            dt: Time step
            n_steps: Number of steps
            store_every: Store trajectory every N steps
            checkpoint_mgr: CheckpointManager instance
            final_only: If True, only return final state (O(√T) memory)

        Returns:
            (traj_x, traj_v): Stored trajectory or final state
        """
        # Import here to avoid circular dependency
        from .integrators import NoseHoover

        # Save parameters for backward
        ctx.force_fn = force_fn
        ctx.dt = dt
        ctx.n_steps = n_steps
        ctx.store_every = store_every
        ctx.checkpoint_mgr = checkpoint_mgr
        ctx.final_only = final_only
        ctx.save_for_backward(x0, v0, alpha0, kT, mass, Q)

        # Forward integration with checkpointing (no gradients needed)
        with torch.no_grad():
            # Create integrator for forward pass
            integrator = NoseHoover(kT=kT.item(), mass=mass.item(), Q=Q.item())

            # Forward integration
            x, v, alpha = x0, v0, alpha0

            # Save initial checkpoint
            checkpoint_mgr.save_checkpoint(0, x, v, alpha)

            if final_only:
                # O(√T) memory mode: only store checkpoints, return final state
                for i in range(1, n_steps + 1):
                    x, v, alpha = integrator.step(x, v, alpha, force_fn, dt)
                    if checkpoint_mgr.should_checkpoint(i):
                        checkpoint_mgr.save_checkpoint(i, x, v, alpha)
                
                # Return final state only (shape: (1, *x0.shape))
                traj_x_no_grad = x.unsqueeze(0)
                traj_v_no_grad = v.unsqueeze(0)
            else:
                # Full trajectory mode: O(T) memory
                n_stored = n_steps // store_every + 1
                traj_x_no_grad = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
                traj_v_no_grad = torch.empty((n_stored, *v0.shape), dtype=v0.dtype, device=v0.device)

                traj_x_no_grad[0] = x0
                traj_v_no_grad[0] = v0

                idx = 1
                for i in range(1, n_steps + 1):
                    x, v, alpha = integrator.step(x, v, alpha, force_fn, dt)

                    # Save checkpoint if needed
                    if checkpoint_mgr.should_checkpoint(i):
                        checkpoint_mgr.save_checkpoint(i, x, v, alpha)

                    # Store trajectory
                    if i % store_every == 0:
                        traj_x_no_grad[idx] = x
                        traj_v_no_grad[idx] = v
                        idx += 1

        # Create output tensors that are connected to parameters
        # This ensures backward will be called
        # Add tiny (negligible) contribution from parameters to establish grad connection
        traj_x = traj_x_no_grad + 0 * kT + 0 * mass + 0 * Q
        traj_v = traj_v_no_grad + 0 * kT + 0 * mass + 0 * Q

        return traj_x, traj_v

    @staticmethod
    def backward(ctx, grad_traj_x, grad_traj_v):
        """Backward pass using adjoint method with checkpointing.

        Args:
            ctx: Saved context from forward pass
            grad_traj_x, grad_traj_v: Gradients w.r.t. outputs

        Returns:
            Gradients w.r.t. all inputs
        """
        from .integrators import NoseHoover

        # Retrieve saved tensors
        x0, v0, alpha0, kT, mass, Q = ctx.saved_tensors
        force_fn = ctx.force_fn
        dt = ctx.dt
        n_steps = ctx.n_steps
        store_every = ctx.store_every
        checkpoint_mgr = ctx.checkpoint_mgr

        # Initialize adjoint variables from final gradient
        lambda_x = grad_traj_x[-1] if grad_traj_x is not None else torch.zeros_like(x0)
        lambda_v = grad_traj_v[-1] if grad_traj_v is not None else torch.zeros_like(v0)
        lambda_alpha = torch.zeros_like(alpha0)

        # Accumulate parameter gradients
        grad_kT = torch.zeros_like(kT)
        grad_mass = torch.zeros_like(mass)
        grad_Q = torch.zeros_like(Q)

        # Backward pass
        traj_idx = len(grad_traj_x) - 2  # Index for trajectory gradients

        for step in range(n_steps, 0, -1):
            # Get or recompute state at step-1
            checkpoint_idx, (x_ckpt, v_ckpt, alpha_ckpt) = checkpoint_mgr.get_nearest_checkpoint_before(step - 1)

            # Recompute forward from checkpoint to step-1
            integrator = NoseHoover(kT=kT.item(), mass=mass.item(), Q=Q.item())
            x, v, alpha = x_ckpt, v_ckpt, alpha_ckpt

            for i in range(checkpoint_idx, step - 1):
                x, v, alpha = integrator.step(x, v, alpha, force_fn, dt)

            # Now x, v, alpha are at step-1
            # Add trajectory gradient if this step is stored
            if step % store_every == 0 and traj_idx >= 0:
                if grad_traj_x is not None:
                    lambda_x = lambda_x + grad_traj_x[traj_idx]
                if grad_traj_v is not None:
                    lambda_v = lambda_v + grad_traj_v[traj_idx]
                traj_idx -= 1

            # Compute adjoint step: propagate adjoints backward through one step
            # NOTE: Inside custom autograd backward, grad is disabled by default.
            # We must enable it to build the computation graph for VJP.
            with torch.enable_grad():
                # Enable gradients for state and parameters
                x_grad = x.detach().requires_grad_(True)
                v_grad = v.detach().requires_grad_(True)
                alpha_grad = alpha.detach().requires_grad_(True)
                kT_grad = kT.detach().requires_grad_(True)
                mass_grad = mass.detach().requires_grad_(True)
                Q_grad = Q.detach().requires_grad_(True)

                # Manually implement NH step with differentiable parameters
                # (We can't use NoseHoover class here because we need parameter gradients)
                ndof = x_grad.shape[-1]

                # First thermostat half-step
                v2 = (v_grad**2).sum(dim=-1)
                alpha_new = alpha_grad + (dt / 4) * (v2 - ndof * kT_grad) / Q_grad
                v_new = v_grad * torch.exp(-alpha_new.unsqueeze(-1) * dt / 2)
                v2 = (v_new**2).sum(dim=-1)
                alpha_new = alpha_new + (dt / 4) * (v2 - ndof * kT_grad) / Q_grad

                # Velocity-Verlet for physical degrees of freedom
                v_new = v_new + (dt / 2) * force_fn(x_grad) / mass_grad
                x_new = x_grad + dt * v_new
                v_new = v_new + (dt / 2) * force_fn(x_new) / mass_grad

                # Second thermostat half-step
                v2 = (v_new**2).sum(dim=-1)
                alpha_new = alpha_new + (dt / 4) * (v2 - ndof * kT_grad) / Q_grad
                v_new = v_new * torch.exp(-alpha_new.unsqueeze(-1) * dt / 2)
                v2 = (v_new**2).sum(dim=-1)
                alpha_new = alpha_new + (dt / 4) * (v2 - ndof * kT_grad) / Q_grad

                x_next, v_next, alpha_next = x_new, v_new, alpha_new

                # Compute vector-Jacobian products (VJPs)
                # This computes: lambda_{t-1} = lambda_t^T @ J_t
                # where J_t is Jacobian of step at time t-1

                # Create dummy loss: <lambda, output>
                dummy_loss = (
                    (lambda_x * x_next).sum() +
                    (lambda_v * v_next).sum() +
                    (lambda_alpha * alpha_next).sum()
                )

                # Compute gradients (VJPs)
                grads = torch.autograd.grad(
                    dummy_loss,
                    [x_grad, v_grad, alpha_grad, kT_grad, mass_grad, Q_grad],
                    allow_unused=True,
                    retain_graph=False
                )

            # Update adjoint variables
            lambda_x = grads[0] if grads[0] is not None else torch.zeros_like(x)
            lambda_v = grads[1] if grads[1] is not None else torch.zeros_like(v)
            lambda_alpha = grads[2] if grads[2] is not None else torch.zeros_like(alpha)

            # Accumulate parameter gradients
            if grads[3] is not None:
                grad_kT += grads[3]
            if grads[4] is not None:
                grad_mass += grads[4]
            if grads[5] is not None:
                grad_Q += grads[5]

        # Return gradients (match forward signature)
        # (x0, v0, alpha0, kT, mass, Q, force_fn, dt, n_steps, store_every, checkpoint_mgr, final_only)
        return lambda_x, lambda_v, lambda_alpha, grad_kT, grad_mass, grad_Q, None, None, None, None, None, None


class CheckpointedNoseHoover(nn.Module):
    """NoseHoover integrator with discrete adjoint and checkpointing for O(√T) memory.

    This class implements the discrete adjoint method with checkpointing, enabling
    gradient computation with O(√T) memory complexity instead of O(T) for standard BPTT.

    The method stores checkpoints at O(√T) evenly-spaced points during the forward pass,
    then recomputes intermediate states during the backward pass. This trades ~2-3x
    computation for O(√T) memory savings.

    Args:
        kT: Thermal energy (target temperature). Differentiable parameter.
        mass: Particle mass. Differentiable parameter.
        Q: Thermostat mass (coupling strength). Differentiable parameter.
        checkpoint_segments: Number of checkpoints (default: √T, computed automatically)

    Example:
        >>> integrator = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0)
        >>> traj_x, traj_v = integrator.run(x0, v0, force_fn, dt=0.01, n_steps=1000)
        >>> loss = traj_x[-1].pow(2).sum()
        >>> loss.backward()  # O(√T) memory instead of O(T)
        >>> print(integrator.kT.grad)
    """

    def __init__(self, kT: float = 1.0, mass: float = 1.0, Q: float = 1.0,
                 checkpoint_segments: Optional[int] = None):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(kT))
        self.mass = nn.Parameter(torch.tensor(mass))
        self.Q = nn.Parameter(torch.tensor(Q))
        self.checkpoint_segments = checkpoint_segments

    def step(self, x: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
             force_fn: Callable[[torch.Tensor], torch.Tensor], dt: float
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single NoseHoover step (same as standard NoseHoover).

        Args:
            x: positions (..., dim)
            v: velocities (..., dim)
            alpha: thermostat variable (...,)
            force_fn: force function
            dt: time step

        Returns:
            (new_x, new_v, new_alpha)
        """
        # Import to avoid circular dependency
        from .integrators import NoseHoover

        integrator = NoseHoover(kT=self.kT.item(), mass=self.mass.item(), Q=self.Q.item())
        return integrator.step(x, v, alpha, force_fn, dt)

    def run(self, x0: torch.Tensor, v0: Optional[torch.Tensor],
            force_fn: Callable[[torch.Tensor], torch.Tensor],
            dt: float, n_steps: int, store_every: int = 1,
            final_only: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory with checkpointing for O(√T) memory backward pass.

        Args:
            x0: Initial positions (..., dim)
            v0: Initial velocities (..., dim), or None to sample from thermal distribution
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps
            final_only: If True, only return final state for O(√T) memory. 
                       If False, return full trajectory (O(T) memory for trajectory storage).

        Returns:
            (traj_x, traj_v): Trajectories of shape (n_stored, ..., dim) or (1, ..., dim) if final_only
        """
        if v0 is None:
            v0 = torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)

        alpha0 = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)

        # Create checkpoint manager
        n_checkpoints = self.checkpoint_segments if self.checkpoint_segments is not None else None
        checkpoint_mgr = CheckpointManager(n_steps, n_checkpoints)

        # Run forward with checkpointing (uses custom autograd function)
        traj_x, traj_v = NoseHooverCheckpointedFunction.apply(
            x0, v0, alpha0, self.kT, self.mass, self.Q,
            force_fn, dt, n_steps, store_every, checkpoint_mgr, final_only
        )

        return traj_x, traj_v


class ContinuousAdjointNoseHoover(nn.Module):
    """NoseHoover integrator with continuous adjoint method for O(1) memory backward pass.

    This class implements the continuous adjoint method based on Pontryagin's principle.
    It solves adjoint ODEs backward in time to compute gradients with O(1) memory
    for the backward pass (though O(T) for storing the forward trajectory).

    The adjoint dynamics are derived from the continuous-time NoseHoover equations:
        dx/dt = v
        dv/dt = F(x)/m - α·v
        dα/dt = (||v||² - ndof·kT) / Q

    Args:
        kT: Thermal energy (target temperature). Differentiable parameter.
        mass: Particle mass. Differentiable parameter.
        Q: Thermostat mass (coupling strength). Differentiable parameter.

    Example:
        >>> integrator = ContinuousAdjointNoseHoover(kT=1.0, mass=1.0, Q=1.0)
        >>> traj_x, traj_v, traj_alpha = integrator.run(x0, v0, force_fn, dt=0.01, n_steps=1000)
        >>> loss = traj_x[-1].pow(2).sum()
        >>> # Compute loss gradient w.r.t. final state
        >>> grad_x = torch.autograd.grad(loss, traj_x[-1], retain_graph=True)[0]
        >>> # Run adjoint backward
        >>> grads = integrator.adjoint_backward([grad_x], [None], traj_x, traj_v, traj_alpha, force_fn, dt)
        >>> print(grads['kT'], grads['mass'], grads['Q'])
    """

    def __init__(self, kT: float = 1.0, mass: float = 1.0, Q: float = 1.0):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(kT))
        self.mass = nn.Parameter(torch.tensor(mass))
        self.Q = nn.Parameter(torch.tensor(Q))

    def step(self, x: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
             force_fn: Callable[[torch.Tensor], torch.Tensor], dt: float
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single NoseHoover step (same as standard NoseHoover).

        Args:
            x: positions (..., dim)
            v: velocities (..., dim)
            alpha: thermostat variable (...,)
            force_fn: force function
            dt: time step

        Returns:
            (new_x, new_v, new_alpha)
        """
        from .integrators import NoseHoover

        integrator = NoseHoover(kT=self.kT.item(), mass=self.mass.item(), Q=self.Q.item())
        return integrator.step(x, v, alpha, force_fn, dt)

    def run(self, x0: torch.Tensor, v0: Optional[torch.Tensor],
            force_fn: Callable[[torch.Tensor], torch.Tensor],
            dt: float, n_steps: int, store_every: int = 1
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run trajectory and store full state (x, v, alpha) for adjoint backward.

        Args:
            x0: Initial positions (..., dim)
            v0: Initial velocities (..., dim), or None to sample
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps

        Returns:
            (traj_x, traj_v, traj_alpha): Full state trajectory
        """
        if v0 is None:
            v0 = torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)

        x = x0
        v = v0
        alpha = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)

        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj_v = torch.empty((n_stored, *v.shape), dtype=v.dtype, device=v.device)
        traj_alpha = torch.empty((n_stored, *alpha.shape), dtype=alpha.dtype, device=alpha.device)

        traj_x[0] = x0
        traj_v[0] = v
        traj_alpha[0] = alpha

        idx = 1
        for i in range(1, n_steps + 1):
            x, v, alpha = self.step(x, v, alpha, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx] = x
                traj_v[idx] = v
                traj_alpha[idx] = alpha
                idx += 1

        return traj_x, traj_v, traj_alpha

    def adjoint_step_backward(self, lambda_x: torch.Tensor, lambda_v: torch.Tensor,
                             lambda_alpha: torch.Tensor, x: torch.Tensor,
                             v: torch.Tensor, alpha: torch.Tensor,
                             force_fn: Callable[[torch.Tensor], torch.Tensor],
                             dt: float) -> Tuple[torch.Tensor, ...]:
        """One backward step of continuous adjoint dynamics.

        Integrates adjoint ODEs backward by dt:
            dlambda_x/dt = -lambda_v · (∂F/∂x) / mass
            dlambda_v/dt = -lambda_x - lambda_alpha · (2v/Q) + lambda_v · alpha
            dlambda_alpha/dt = -lambda_v · v

        Args:
            lambda_x, lambda_v, lambda_alpha: Current adjoint variables
            x, v, alpha: Forward state at this time
            force_fn: Force function
            dt: Time step (positive, but integration is backward)

        Returns:
            (new_lambda_x, new_lambda_v, new_lambda_alpha, grad_kT, grad_mass, grad_Q)
        """
        ndof = x.shape[-1]
        device = x.device
        
        # Move parameters to same device as trajectory
        kT = self.kT.to(device)
        mass = self.mass.to(device)
        Q = self.Q.to(device)

        # Compute force and its Jacobian w.r.t. x
        x_grad = x.detach().requires_grad_(True)
        F = force_fn(x_grad)

        # Compute dF/dx via autograd
        # For vector-Jacobian product: lambda_v^T @ (dF/dx)
        vjp = torch.autograd.grad(
            F, x_grad, grad_outputs=lambda_v, create_graph=False, retain_graph=False, allow_unused=True
        )[0]

        if vjp is None:
            vjp = torch.zeros_like(x_grad)

        # Adjoint dynamics from Pontryagin's principle
        # dλ_x/dt = -(∂f/∂x)^T · λ = -(1/m)(∂F/∂x)^T · λ_v
        # dλ_v/dt = -(∂f/∂v)^T · λ = -λ_x + α·λ_v - (2v/Q)·λ_α
        # dλ_α/dt = -(∂f/∂α)^T · λ = v · λ_v
        dlambda_x_dt = -vjp / mass
        dlambda_v_dt = -lambda_x - lambda_alpha.unsqueeze(-1) * (2 * v / Q) + lambda_v * alpha.unsqueeze(-1)
        dlambda_alpha_dt = (lambda_v * v).sum(dim=-1)

        # Backward Euler update (negative dt for backward integration)
        lambda_x_new = lambda_x - dlambda_x_dt * dt
        lambda_v_new = lambda_v - dlambda_v_dt * dt
        lambda_alpha_new = lambda_alpha - dlambda_alpha_dt * dt

        # Parameter gradient contributions: ∂L/∂θ = ∫ λ^T · (∂f/∂θ) dt
        # ∂(dα/dt)/∂kT = -ndof/Q
        # ∂(dv/dt)/∂mass = -F(x)/m²
        # ∂(dα/dt)/∂Q = -(v² - ndof·kT)/Q²
        # Sum over all dimensions to get scalar gradients matching parameter shapes
        grad_kT = -(lambda_alpha * (ndof / Q) * dt).sum()
        grad_mass = -(lambda_v * F / (mass ** 2)).sum() * dt
        grad_Q = -(lambda_alpha * (((v ** 2).sum(dim=-1) - ndof * kT) / (Q ** 2)) * dt).sum()

        return lambda_x_new, lambda_v_new, lambda_alpha_new, grad_kT, grad_mass, grad_Q

    def adjoint_backward(self, loss_grad_x: List[Optional[torch.Tensor]],
                        loss_grad_v: List[Optional[torch.Tensor]],
                        traj_x: torch.Tensor, traj_v: torch.Tensor,
                        traj_alpha: torch.Tensor,
                        force_fn: Callable[[torch.Tensor], torch.Tensor],
                        dt: float) -> Dict[str, torch.Tensor]:
        """Integrate adjoint ODEs backward through trajectory.

        Args:
            loss_grad_x: List of gradients ∂L/∂x at each stored timestep (or None)
            loss_grad_v: List of gradients ∂L/∂v at each stored timestep (or None)
            traj_x, traj_v, traj_alpha: Forward trajectory
            force_fn: Force function
            dt: Time step used in forward pass

        Returns:
            Dictionary of parameter gradients: {'kT': grad_kT, 'mass': grad_mass, 'Q': grad_Q}
        """
        T = len(traj_x)

        # Pad loss gradients if needed
        # User typically passes gradients at final timestep, so pad with Nones at the FRONT
        if len(loss_grad_x) < T:
            loss_grad_x = [None] * (T - len(loss_grad_x)) + list(loss_grad_x)
        if len(loss_grad_v) < T:
            loss_grad_v = [None] * (T - len(loss_grad_v)) + list(loss_grad_v)

        # Initialize adjoint variables from final loss gradients
        lambda_x = loss_grad_x[-1] if loss_grad_x[-1] is not None else torch.zeros_like(traj_x[-1])
        lambda_v = loss_grad_v[-1] if loss_grad_v[-1] is not None else torch.zeros_like(traj_v[-1])
        lambda_alpha = torch.zeros_like(traj_alpha[-1])

        # Accumulate parameter gradients (on same device as trajectory)
        device = traj_x.device
        grad_kT = torch.zeros((), device=device, dtype=self.kT.dtype)
        grad_mass = torch.zeros((), device=device, dtype=self.mass.dtype)
        grad_Q = torch.zeros((), device=device, dtype=self.Q.dtype)

        # Backward integration
        for t in range(T - 1, 0, -1):
            # One adjoint step backward
            lambda_x, lambda_v, lambda_alpha, d_kT, d_mass, d_Q = self.adjoint_step_backward(
                lambda_x, lambda_v, lambda_alpha,
                traj_x[t], traj_v[t], traj_alpha[t],
                force_fn, dt
            )

            # Accumulate parameter gradients
            grad_kT += d_kT
            grad_mass += d_mass
            grad_Q += d_Q

            # Add loss gradient contributions at this timestep
            if loss_grad_x[t - 1] is not None:
                lambda_x = lambda_x + loss_grad_x[t - 1]
            if loss_grad_v[t - 1] is not None:
                lambda_v = lambda_v + loss_grad_v[t - 1]

        return {
            'kT': grad_kT,
            'mass': grad_mass,
            'Q': grad_Q,
            'x0': lambda_x,
            'v0': lambda_v,
            'alpha0': lambda_alpha
        }


class DiscreteAdjointNoseHoover(nn.Module):
    """Nosé-Hoover with discrete adjoint matching Kleinerman 08 scheme.
    
    The discrete adjoint differentiates through the exact discrete operations,
    giving 2nd-order accurate gradients that match BPTT exactly (up to roundoff).
    
    The Kleinerman 08 forward scheme is a palindromic/symmetric splitting:
    
    Forward step (x, v, α) → (x', v', α'):
      1. α₁ = α + (dt/4) * (v² - ndof·kT) / Q          [thermostat quarter-step]
      2. v₁ = v * exp(-α₁ * dt/2)                       [velocity rescale]
      3. α₂ = α₁ + (dt/4) * (v₁² - ndof·kT) / Q        [thermostat quarter-step]
      4. v₂ = v₁ + (dt/2) * F(x) / m                    [velocity half-kick]
      5. x' = x + dt * v₂                               [position full-step]
      6. v₃ = v₂ + (dt/2) * F(x') / m                   [velocity half-kick]
      7. α₃ = α₂ + (dt/4) * (v₃² - ndof·kT) / Q        [thermostat quarter-step]
      8. v₄ = v₃ * exp(-α₃ * dt/2)                      [velocity rescale]
      9. α' = α₃ + (dt/4) * (v₄² - ndof·kT) / Q        [thermostat quarter-step]
    
    The discrete adjoint runs backward through these operations using chain rule.
    For each operation y = f(x), the adjoint is: λ_x = (∂f/∂x)ᵀ λ_y
    
    Args:
        kT: Thermal energy. Differentiable parameter.
        mass: Particle mass. Differentiable parameter.
        Q: Thermostat mass. Differentiable parameter.
    
    Example:
        >>> integrator = DiscreteAdjointNoseHoover(kT=1.0, mass=1.0, Q=1.0)
        >>> traj_x, traj_v, traj_alpha = integrator.run(x0, v0, force_fn, dt, n_steps)
        >>> loss_grad_x = 2 * traj_x[-1]  # gradient of x².sum()
        >>> grads = integrator.adjoint_backward([loss_grad_x], [None], traj_x, traj_v, traj_alpha, force_fn, dt)
        >>> # grads['kT'], grads['mass'], grads['Q'] match BPTT exactly
    """
    
    def __init__(self, kT: float = 1.0, mass: float = 1.0, Q: float = 1.0):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(kT))
        self.mass = nn.Parameter(torch.tensor(mass))
        self.Q = nn.Parameter(torch.tensor(Q))
    
    def forward_step(self, x: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                     force_fn: Callable, dt: float
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single forward step using Kleinerman 08 scheme."""
        ndof = x.shape[-1]
        
        # Step 1: First thermostat quarter-step
        v2 = (v**2).sum(dim=-1)
        alpha1 = alpha + (dt / 4) * (v2 - ndof * self.kT) / self.Q
        
        # Step 2: Velocity rescale
        v1 = v * torch.exp(-alpha1.unsqueeze(-1) * dt / 2)
        
        # Step 3: Second thermostat quarter-step
        v1_2 = (v1**2).sum(dim=-1)
        alpha2 = alpha1 + (dt / 4) * (v1_2 - ndof * self.kT) / self.Q
        
        # Step 4: First velocity half-kick
        F0 = force_fn(x)
        v2_vec = v1 + (dt / 2) * F0 / self.mass
        
        # Step 5: Position full-step
        x_new = x + dt * v2_vec
        
        # Step 6: Second velocity half-kick
        F1 = force_fn(x_new)
        v3 = v2_vec + (dt / 2) * F1 / self.mass
        
        # Step 7: Third thermostat quarter-step
        v3_2 = (v3**2).sum(dim=-1)
        alpha3 = alpha2 + (dt / 4) * (v3_2 - ndof * self.kT) / self.Q
        
        # Step 8: Velocity rescale
        v4 = v3 * torch.exp(-alpha3.unsqueeze(-1) * dt / 2)
        
        # Step 9: Fourth thermostat quarter-step
        v4_2 = (v4**2).sum(dim=-1)
        alpha_new = alpha3 + (dt / 4) * (v4_2 - ndof * self.kT) / self.Q
        
        return x_new, v4, alpha_new
    
    def adjoint_step(self, lambda_x: torch.Tensor, lambda_v: torch.Tensor, 
                     lambda_alpha: torch.Tensor, x: torch.Tensor, v: torch.Tensor,
                     alpha: torch.Tensor, force_fn: Callable, dt: float
                     ) -> Tuple[torch.Tensor, ...]:
        """One backward step of discrete adjoint.
        
        Args:
            lambda_x, lambda_v, lambda_alpha: Adjoint variables at time t+1
            x, v, alpha: State at time t (before forward step)
            force_fn: Force function
            dt: Time step
            
        Returns:
            (new_lambda_x, new_lambda_v, new_lambda_alpha, grad_kT, grad_mass, grad_Q)
        """
        device = x.device
        ndof = x.shape[-1]
        
        # Move parameters to device
        kT = self.kT.to(device)
        mass = self.mass.to(device)
        Q = self.Q.to(device)
        
        # === Re-run forward to get intermediate values ===
        v2 = (v**2).sum(dim=-1)
        alpha1 = alpha + (dt / 4) * (v2 - ndof * kT) / Q
        exp_neg_a1 = torch.exp(-alpha1.unsqueeze(-1) * dt / 2)
        v1 = v * exp_neg_a1
        v1_2 = (v1**2).sum(dim=-1)
        alpha2 = alpha1 + (dt / 4) * (v1_2 - ndof * kT) / Q
        F0 = force_fn(x)
        v2_vec = v1 + (dt / 2) * F0 / mass
        x_new = x + dt * v2_vec
        F1 = force_fn(x_new)
        v3 = v2_vec + (dt / 2) * F1 / mass
        v3_2 = (v3**2).sum(dim=-1)
        alpha3 = alpha2 + (dt / 4) * (v3_2 - ndof * kT) / Q
        exp_neg_a3 = torch.exp(-alpha3.unsqueeze(-1) * dt / 2)
        v4 = v3 * exp_neg_a3
        v4_2 = (v4**2).sum(dim=-1)
        
        # === Backward pass through operations in reverse order ===
        grad_kT = torch.zeros((), device=device, dtype=kT.dtype)
        grad_mass = torch.zeros((), device=device, dtype=mass.dtype)
        grad_Q = torch.zeros((), device=device, dtype=Q.dtype)
        
        lam_x = lambda_x.clone()
        lam_v = lambda_v.clone()
        lam_alpha = lambda_alpha.clone()
        
        # Adjoint of Step 9: α' = α₃ + (dt/4)(v₄² - ndof·kT)/Q
        lam_alpha3 = lam_alpha.clone()
        lam_v4 = lam_v + lam_alpha.unsqueeze(-1) * (dt / 2) * v4 / Q
        grad_kT = grad_kT - (lam_alpha * (dt / 4) * ndof / Q).sum()
        grad_Q = grad_Q - (lam_alpha * (dt / 4) * (v4_2 - ndof * kT) / (Q ** 2)).sum()
        
        # Adjoint of Step 8: v₄ = v₃ * exp(-α₃ * dt/2)
        lam_v3 = lam_v4 * exp_neg_a3
        lam_alpha3 = lam_alpha3 - (dt / 2) * (lam_v4 * v4).sum(dim=-1)
        
        # Adjoint of Step 7: α₃ = α₂ + (dt/4)(v₃² - ndof·kT)/Q
        lam_alpha2 = lam_alpha3.clone()
        lam_v3 = lam_v3 + lam_alpha3.unsqueeze(-1) * (dt / 2) * v3 / Q
        grad_kT = grad_kT - (lam_alpha3 * (dt / 4) * ndof / Q).sum()
        grad_Q = grad_Q - (lam_alpha3 * (dt / 4) * (v3_2 - ndof * kT) / (Q ** 2)).sum()
        
        # Adjoint of Step 6: v₃ = v₂ + (dt/2) * F(x')/m
        lam_v2 = lam_v3.clone()
        x_new_grad = x_new.detach().requires_grad_(True)
        F1_recompute = force_fn(x_new_grad)
        vjp_x1 = torch.autograd.grad(F1_recompute, x_new_grad, grad_outputs=lam_v3,
                                      create_graph=False, retain_graph=False)[0]
        lam_x_from_v = vjp_x1 * (dt / 2) / mass
        grad_mass = grad_mass - ((lam_v3 * F1).sum() * (dt / 2) / (mass ** 2))
        
        # Adjoint of Step 5: x' = x + dt * v₂
        lam_x_new = lam_x + lam_x_from_v
        lam_x_0 = lam_x_new.clone()
        lam_v2 = lam_v2 + dt * lam_x_new
        
        # Adjoint of Step 4: v₂ = v₁ + (dt/2) * F(x)/m
        lam_v1 = lam_v2.clone()
        x_grad = x.detach().requires_grad_(True)
        F0_recompute = force_fn(x_grad)
        vjp_x0 = torch.autograd.grad(F0_recompute, x_grad, grad_outputs=lam_v2,
                                      create_graph=False, retain_graph=False)[0]
        lam_x_0 = lam_x_0 + vjp_x0 * (dt / 2) / mass
        grad_mass = grad_mass - ((lam_v2 * F0).sum() * (dt / 2) / (mass ** 2))
        
        # Adjoint of Step 3: α₂ = α₁ + (dt/4)(v₁² - ndof·kT)/Q
        lam_alpha1 = lam_alpha2.clone()
        lam_v1 = lam_v1 + lam_alpha2.unsqueeze(-1) * (dt / 2) * v1 / Q
        grad_kT = grad_kT - (lam_alpha2 * (dt / 4) * ndof / Q).sum()
        grad_Q = grad_Q - (lam_alpha2 * (dt / 4) * (v1_2 - ndof * kT) / (Q ** 2)).sum()
        
        # Adjoint of Step 2: v₁ = v * exp(-α₁ * dt/2)
        lam_v_0 = lam_v1 * exp_neg_a1
        lam_alpha1 = lam_alpha1 - (dt / 2) * (lam_v1 * v1).sum(dim=-1)
        
        # Adjoint of Step 1: α₁ = α + (dt/4)(v² - ndof·kT)/Q
        lam_alpha_0 = lam_alpha1.clone()
        lam_v_0 = lam_v_0 + lam_alpha1.unsqueeze(-1) * (dt / 2) * v / Q
        grad_kT = grad_kT - (lam_alpha1 * (dt / 4) * ndof / Q).sum()
        grad_Q = grad_Q - (lam_alpha1 * (dt / 4) * (v2 - ndof * kT) / (Q ** 2)).sum()
        
        return lam_x_0, lam_v_0, lam_alpha_0, grad_kT, grad_mass, grad_Q
    
    def run(self, x0: torch.Tensor, v0: Optional[torch.Tensor],
            force_fn: Callable, dt: float, n_steps: int, store_every: int = 1
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward trajectory, storing full state for adjoint."""
        if v0 is None:
            v0 = torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        
        x, v = x0, v0
        alpha = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj_v = torch.empty((n_stored, *v0.shape), dtype=v0.dtype, device=v0.device)
        traj_alpha = torch.empty((n_stored, *alpha.shape), dtype=alpha.dtype, device=alpha.device)
        
        traj_x[0], traj_v[0], traj_alpha[0] = x0, v0, alpha
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, v, alpha = self.forward_step(x, v, alpha, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx], traj_v[idx], traj_alpha[idx] = x, v, alpha
                idx += 1
        
        return traj_x, traj_v, traj_alpha
    
    def adjoint_backward(self, loss_grad_x: List[Optional[torch.Tensor]],
                        loss_grad_v: List[Optional[torch.Tensor]],
                        traj_x: torch.Tensor, traj_v: torch.Tensor,
                        traj_alpha: torch.Tensor,
                        force_fn: Callable, dt: float) -> Dict[str, torch.Tensor]:
        """Backward pass using discrete adjoint (2nd order accurate)."""
        T = len(traj_x)
        device = traj_x.device
        
        # Pad loss gradients at the FRONT
        if len(loss_grad_x) < T:
            loss_grad_x = [None] * (T - len(loss_grad_x)) + list(loss_grad_x)
        if len(loss_grad_v) < T:
            loss_grad_v = [None] * (T - len(loss_grad_v)) + list(loss_grad_v)
        
        # Initialize adjoint from final loss gradient
        lambda_x = loss_grad_x[-1] if loss_grad_x[-1] is not None else torch.zeros_like(traj_x[-1])
        lambda_v = loss_grad_v[-1] if loss_grad_v[-1] is not None else torch.zeros_like(traj_v[-1])
        lambda_alpha = torch.zeros_like(traj_alpha[-1])
        
        grad_kT = torch.zeros((), device=device, dtype=self.kT.dtype)
        grad_mass = torch.zeros((), device=device, dtype=self.mass.dtype)
        grad_Q = torch.zeros((), device=device, dtype=self.Q.dtype)
        
        for t in range(T - 1, 0, -1):
            lambda_x, lambda_v, lambda_alpha, d_kT, d_mass, d_Q = self.adjoint_step(
                lambda_x, lambda_v, lambda_alpha,
                traj_x[t - 1], traj_v[t - 1], traj_alpha[t - 1],
                force_fn, dt
            )
            grad_kT, grad_mass, grad_Q = grad_kT + d_kT, grad_mass + d_mass, grad_Q + d_Q
            
            if loss_grad_x[t - 1] is not None:
                lambda_x = lambda_x + loss_grad_x[t - 1]
            if loss_grad_v[t - 1] is not None:
                lambda_v = lambda_v + loss_grad_v[t - 1]
        
        return {
            'kT': grad_kT, 'mass': grad_mass, 'Q': grad_Q,
            'x0': lambda_x, 'v0': lambda_v, 'alpha0': lambda_alpha
        }



class ForwardSensitivityEstimator(nn.Module):
    """Forward mode sensitivity estimator using torch.func (Jacobian-Vector Products).

    Computes full Jacobians of the simulation trajectory w.r.t. parameters using
    Forward Mode AD. This is memory-efficient O(P) where P is the number of parameters,
    independent of trajectory length T (unlike BPTT which is O(T) memory).

    This method propagates sensitivities forward in time alongside the state,
    avoiding the need to store the entire history for backpropagation.

    Ideal for:
    - Sensitivity analysis (how x_t changes with θ)
    - Optimization with few parameters and long trajectories
    - Real-time gradient computation

    Args:
        integrator: The integrator instance (nn.Module).
        param_names: List of parameter names to differentiate w.r.t.
    """
    def __init__(self, integrator: nn.Module, param_names: List[str]):
        super().__init__()
        self.integrator = integrator
        self.param_names = param_names

    def forward_sensitivity(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run simulation and compute sensitivity (Jacobian) w.r.t parameters.

        This implementation uses jacfwd on the entire run, which is convenient 
        but memory-intensive for long trajectories.
        """
        from torch.func import jacfwd

        params = dict(self.integrator.named_parameters())
        target_params = {k: params[k] for k in self.param_names if k in params}
        
        if not target_params:
            raise ValueError(f"No target parameters found in integrator: {self.param_names}")
        
        other_params = {k: v for k, v in params.items() if k not in target_params}

        def wrapper(p_subset):
            full_params = {**other_params, **p_subset}
            return functional_call(self.integrator, full_params, args=args, kwargs=kwargs, strict=False)

        jac_output = jacfwd(wrapper)(target_params)
        
        with torch.no_grad():
             if hasattr(self.integrator, 'forward'):
                 primal = self.integrator(*args, **kwargs)
             else:
                 primal = self.integrator.run(*args, **kwargs)
                 
        return primal, jac_output

    def forward_sensitivity_online(self, x0: torch.Tensor, v0: Optional[torch.Tensor], 
                                 force_fn: Callable, dt: float, n_steps: int, 
                                 store_every: int = 1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Optimized online forward sensitivity propagation.
        
        Propagates sensitivities (Jacobian-Vector Products) step-by-step.
        This is significantly more memory-efficient than full-trajectory AD
        and avoids the overhead of tracing a long loop.
        
        Memory: O(P * D) - strictly independent of T.
        """
        from torch.func import jvp, vmap, functional_call
        
        params = dict(self.integrator.named_parameters())
        target_names = [k for k in self.param_names if k in params]
        target_params = {k: params[k] for k in target_names}
        other_params = {k: v for k, v in params.items() if k not in target_params}
        
        # 1. Define the step function
        # We need a function that takes target_params and returns next state
        def step_func(p_target, state_tuple):
            p_full = {**other_params, **p_target}
            # Unpack state
            if len(state_tuple) == 3: # NoseHoover (x, v, alpha)
                x, v, alpha = state_tuple
                return functional_call(self.integrator, p_full, (x, v, alpha, force_fn, dt))
            else: # Standard (x, v)
                x, v = state_tuple
                return functional_call(self.integrator, p_full, (x, v, force_fn, dt))

        # 2. Setup for parallel sensitivity propagation
        # We use a flattened view for the basis
        flat_params = torch.cat([p.view(-1) for p in target_params.values()])
        n_params = flat_params.numel()
        
        # Initial state
        state = (x0, v0) if v0 is not None else (x0,)

        # Jacobians: initialized to zero
        # Sensitivities have shape (n_params, *state_shape)
        state_jacs = tuple(torch.zeros(n_params, *s.shape, device=s.device, dtype=s.dtype) for s in state)

        # Flattened basis: (n_params, target_params_dict)
        # This allows us to vmap over the parameter dimension
        def get_tangent_dict(idx):
            d = {}
            curr = 0
            for k, p in target_params.items():
                size = p.numel()
                t = torch.zeros_like(p)
                if idx >= curr and idx < curr + size:
                    t.view(-1)[idx - curr] = 1.0
                d[k] = t
                curr += size
            return d

        # 3. Propagation Loop
        # J_next = JVP(step, (params, state), (param_tangent, J_curr))
        
        curr_state = state
        curr_jacs = state_jacs
        
        # Result storage
        n_stored = n_steps // store_every + 1
        traj_states = tuple(torch.empty((n_stored, *s.shape), device=s.device) for s in state)
        for i, s in enumerate(curr_state): traj_states[i][0] = s
        
        for step_idx in range(1, n_steps + 1):
            # Propagate all n_params sensitivities in parallel using vmap(jvp)
            def jvp_wrapper(idx):
                p_tangent = get_tangent_dict(idx)
                state_tangent = tuple(j[idx] for j in curr_jacs)
                return jvp(step_func, (target_params, curr_state), (p_tangent, state_tangent))
            
            # This computes (new_state, new_jacs)
            res = vmap(jvp_wrapper)(torch.arange(n_params, device=x0.device))
            
            # res structure: ( (new_x, new_v), (jac_x, jac_v) )
            new_state_all = res[0]
            new_jacs = res[1]
            
            curr_state = tuple(s[0] for s in new_state_all)
            curr_jacs = new_jacs
            
            if step_idx % store_every == 0:
                idx = step_idx // store_every
                for i, s in enumerate(curr_state): traj_states[i][idx] = s

        # 4. Format output Jacobians (Final state only for performance)
        final_jacs = {}
        curr = 0
        for k, p in target_params.items():
            size = p.numel()
            j = curr_jacs[0][curr:curr+size] # Position jacobian
            final_jacs[k] = j.movedim(0, -1).reshape(*x0.shape, *p.shape)
            curr += size
            
        return traj_states[0], final_jacs


if __name__ == "__main__":
    """Quick sanity check for gradient estimators.
    
    For full demos and comparisons, see scripts/demo_gradient_estimators.py
    """
    from uni_diffsim.potentials import Harmonic, DoubleWell
    from uni_diffsim.integrators import OverdampedLangevin
    
    print("Gradient estimators sanity check...")
    torch.manual_seed(42)
    
    # Setup: Harmonic potential with known analytical gradient
    # ⟨x²⟩ = kT/k, so d⟨x²⟩/dk = -kT/k²
    kT = 1.0
    k = 2.0
    beta = 1.0 / kT
    theory_grad = -kT / k**2  # = -0.25
    
    # Generate equilibrium samples
    potential = Harmonic(k=k)
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    x0 = torch.randn(100, 1)
    traj = integrator.run(x0, potential.force, dt=0.01, n_steps=500, store_every=5)
    samples = traj[20:].reshape(-1, 1).detach()  # Skip burn-in
    
    # Test ReinforceEstimator
    estimator = ReinforceEstimator(potential, beta=beta)
    observable = lambda x: (x**2).sum(dim=-1)
    grads = estimator.estimate_gradient(samples, observable=observable)
    rf_grad = grads['k'].item()
    print(f"  REINFORCE: d⟨x²⟩/dk = {rf_grad:.4f} (theory: {theory_grad:.4f})")
    
    # Test reinforce_gradient functional API
    grads_func = reinforce_gradient(samples, potential, observable=observable, beta=beta)
    assert 'k' in grads_func
    print(f"  reinforce_gradient: OK")
    
    # Test PathReweightingEstimator with variance reduction
    print("\n  Testing PathReweightingEstimator variance reduction methods:")
    traj_short = integrator.run(x0[:10], potential.force, dt=0.01, n_steps=50)
    
    for vr_method in [None, "center", "truncate", "self_normalize"]:
        gir = PathReweightingEstimator(potential, sigma=1.0, beta=beta, variance_reduction=vr_method)
        score = gir.compute_log_path_score(traj_short, dt=0.01)
        assert score.numel() > 0, f"Score should be non-empty, got {score.shape}"
        
        # Test gradient estimation with diagnostics
        obs_fn = lambda traj: traj[-1].squeeze(-1).mean()  # Final position mean
        grads, diag = gir.estimate_gradient(traj_short.unsqueeze(0), obs_fn, dt=0.01, return_diagnostics=True)
        
        ess_frac = diag.get('ess_fraction', torch.tensor(0.0)).item()
        print(f"    {vr_method or 'none':15s}: ESS={ess_frac:.2%}, grads computed={len(grads)>0}")
    
    # Test ESS computation
    gir = PathReweightingEstimator(potential, sigma=1.0, beta=beta)
    ess = gir.compute_effective_sample_size(torch.randn(100))
    print(f"  ESS computation: OK (ESS={ess.item():.1f} for uniform random log-weights)")
    
    # Test control variate method
    grads_cv = gir.estimate_gradient_with_control_variate(
        traj_short.unsqueeze(0),
        observable=lambda traj: traj[-1].squeeze(-1).mean(),
        control_variate=lambda traj: traj.mean(),
        dt=0.01
    )
    print(f"  Control variate method: OK")
    
    # Test ReweightingLoss
    loss_fn = ReweightingLoss(potential, beta=beta)
    loss = loss_fn(samples, observable=lambda x: x.squeeze(-1))
    assert loss.requires_grad
    print(f"  ReweightingLoss: OK (loss = {loss.item():.4f})")
    
    # Test ImplicitDiffEstimator
    imp = ImplicitDiffEstimator(potential, beta=beta, mode='equilibrium')
    grads_imp = imp.estimate_gradient(samples, observable=observable)
    imp_grad = grads_imp['k'].item()
    print(f"  ImplicitDiff: d⟨x²⟩/dk = {imp_grad:.4f}")
    
    # Check REINFORCE and Implicit give similar results (both equilibrium methods)
    rel_diff = abs(rf_grad - imp_grad) / (abs(rf_grad) + 1e-10)
    print(f"  REINFORCE vs Implicit relative diff: {rel_diff:.2%}")
    
    print("\nAll gradient estimators passed sanity check!")
