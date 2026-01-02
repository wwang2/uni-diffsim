"""Integrators for molecular dynamics.

All integrators are fully vectorized and support batch dimensions.
Convention: positions x have shape (..., dim), velocities v have shape (..., dim).

Integrators:
- OverdampedLangevin: High-friction stochastic dynamics
- BAOAB: Underdamped Langevin with excellent sampling
- VelocityVerlet: Symplectic NVE integrator
- NoseHooverChain: Deterministic thermostat
- ESH: Energy Sampling Hamiltonian (deterministic ergodic sampling)
- GLE: Generalized Langevin with colored noise

All integrator parameters (gamma, kT, mass, etc.) are registered as nn.Parameters
for automatic differentiation. Gradients w.r.t. these parameters can be computed
via backpropagation through the trajectory.
"""

import torch
import torch.nn as nn
from typing import Callable
import math


ForceFunc = Callable[[torch.Tensor], torch.Tensor]
GradFunc = Callable[[torch.Tensor], torch.Tensor]  # For ESH: gradient of energy


class Integrator(nn.Module):
    """Base class for all integrators.

    Provides the common run loop logic via _integrate method.
    """

    def forward(self, *args, **kwargs):
        """Alias for run() to support functional calls."""
        return self.run(*args, **kwargs)

    def _integrate(self, state: tuple[torch.Tensor, ...], step_fn: Callable,
                   n_steps: int, store_every: int, final_only: bool,
                   store_indices: list[int] | None = None) -> tuple[torch.Tensor, ...]:
        """Generic integration loop.

        Args:
            state: Initial state tuple (e.g. (x,) or (x, v)).
            step_fn: Function that takes unpacked state and returns new unpacked state.
            n_steps: Number of integration steps.
            store_every: Store interval.
            final_only: If True, only return final state.
            store_indices: Indices of state variables to store in trajectory.
                           If None, stores all variables.

        Returns:
            Tuple of trajectories or final states.
        """
        if store_indices is None:
            store_indices = range(len(state))

        if final_only:
            for _ in range(n_steps):
                state = step_fn(*state)
                if not isinstance(state, tuple):
                    state = (state,)

            # Return final state with added batch dim
            results = tuple(state[i].unsqueeze(0) for i in store_indices)
            return results

        # Prepare trajectory storage
        n_stored = n_steps // store_every + 1
        trajs = [torch.empty((n_stored, *state[i].shape), dtype=state[i].dtype, device=state[i].device)
                 for i in store_indices]

        # Store initial state
        for i, traj in enumerate(trajs):
            traj[0] = state[store_indices[i]]

        idx = 1
        for i in range(1, n_steps + 1):
            state = step_fn(*state)
            if not isinstance(state, tuple):
                state = (state,)

            if i % store_every == 0:
                for j, traj in enumerate(trajs):
                    traj[idx] = state[store_indices[j]]
                idx += 1

        return tuple(trajs)


class OverdampedLangevin(Integrator):
    """Overdamped Langevin dynamics: dx = F/γ dt + √(2kT/γ) dW.
    
    High-friction limit where inertia is negligible.
    Samples from Boltzmann distribution p(x) ∝ exp(-U(x)/kT).
    
    Args:
        gamma: Friction coefficient. Differentiable parameter.
        kT: Thermal energy (temperature × Boltzmann constant). Differentiable parameter.
    """
    
    def __init__(self, gamma: float = 1.0, kT: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.kT = nn.Parameter(torch.tensor(kT))
    
    def step(self, x: torch.Tensor, force_fn: ForceFunc, dt: float | torch.Tensor) -> torch.Tensor:
        """Single integration step. Returns new positions."""
        force = force_fn(x)
        noise_scale = torch.sqrt(2 * self.kT * dt / self.gamma)
        return x + (force / self.gamma) * dt + noise_scale * torch.randn_like(x)

    def run(self, x0: torch.Tensor, force_fn: ForceFunc, dt: float | torch.Tensor,
            n_steps: int, store_every: int = 1, final_only: bool = False) -> torch.Tensor:
        """Run trajectory. Returns (n_stored, ..., dim) positions.
        
        Args:
            x0: Initial positions (..., dim)
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps (ignored if final_only=True)
            final_only: If True, only return final state with shape (1, ..., dim)
            
        Returns:
            Trajectory of shape (n_stored, ..., dim) or (1, ..., dim) if final_only
        """
        state = (x0,)
        step_fn = lambda x: self.step(x, force_fn, dt)
        
        result = self._integrate(state, step_fn, n_steps, store_every, final_only)
        return result[0]


class PreconditionedOverdampedLangevin(Integrator):
    """Overdamped Langevin dynamics with anisotropic mass/mobility matrix.

    dx = -M⁻¹∇U(x) dt + √(2kT M⁻¹) dW

    We parameterize M⁻¹ (mobility) as D = R Rᵀ to ensure positive definiteness.
    R is a learnable parameter matrix.

    Args:
        dim: Dimension of the system.
        kT: Thermal energy. Differentiable parameter.
        R: Preconditioning matrix factor. If None, initialized to identity.
           Shape (dim, dim). Differentiable parameter.
    """

    def __init__(self, dim: int, kT: float = 1.0, R: torch.Tensor | None = None):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(kT))
        self.dim = dim

        if R is None:
            R = torch.eye(dim)

        if R.shape != (dim, dim):
            raise ValueError(f"R must be shape ({dim}, {dim}), got {R.shape}")

        self.R = nn.Parameter(R)

    @property
    def diffusion_matrix(self) -> torch.Tensor:
        """Mobility/Diffusion matrix D = R Rᵀ (equivalent to M⁻¹)."""
        return self.R @ self.R.T

    @property
    def mass_matrix(self) -> torch.Tensor:
        """Mass/Friction matrix M = (R Rᵀ)⁻¹."""
        return torch.linalg.inv(self.diffusion_matrix)

    @property
    def inv_mass_matrix(self) -> torch.Tensor:
        """Inverse mass matrix M⁻¹ = R Rᵀ."""
        return self.diffusion_matrix

    def step(self, x: torch.Tensor, force_fn: ForceFunc, dt: float | torch.Tensor) -> torch.Tensor:
        """Single integration step. Returns new positions."""
        force = force_fn(x)  # (..., dim)

        # Drift: D * F = (R Rᵀ) F
        # Note: force is (..., dim). We treat it as a column vector per sample for matrix mult.
        # But in PyTorch (..., dim) is row vectors.
        # drift = (D @ force.T).T = force @ D.T = force @ D (since D symmetric)
        D = self.diffusion_matrix
        drift = force @ D

        # Noise: √(2kT) * R * ξ
        # ξ ~ N(0, I)
        # noise = √(2kT) * (R @ ξ.T).T = √(2kT) * ξ @ R.T
        xi = torch.randn_like(x)
        noise_scale = torch.sqrt(2 * self.kT * dt)
        noise = noise_scale * (xi @ self.R.T)

        return x + drift * dt + noise

    def run(self, x0: torch.Tensor, force_fn: ForceFunc, dt: float | torch.Tensor,
            n_steps: int, store_every: int = 1, final_only: bool = False) -> torch.Tensor:
        """Run trajectory. Returns (n_stored, ..., dim)."""
        state = (x0,)
        step_fn = lambda x: self.step(x, force_fn, dt)

        result = self._integrate(state, step_fn, n_steps, store_every, final_only)
        return result[0]


class BAOAB(Integrator):
    """BAOAB splitting for underdamped Langevin dynamics.
    
    B: velocity kick (half), A: position drift (half), O: Ornstein-Uhlenbeck noise,
    A: position drift (half), B: velocity kick (half).
    
    Excellent sampling properties with low discretization error.
    
    Args:
        gamma: Friction coefficient. Differentiable parameter.
        kT: Thermal energy. Differentiable parameter.
        mass: Particle mass. Differentiable parameter.
    """
    
    def __init__(self, gamma: float = 1.0, kT: float = 1.0, mass: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.kT = nn.Parameter(torch.tensor(kT))
        self.mass = nn.Parameter(torch.tensor(mass))
    
    def step(self, x: torch.Tensor, v: torch.Tensor, force_fn: ForceFunc, 
             dt: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single BAOAB step. Returns (new_x, new_v)."""
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + (dt / 2) * v
        alpha = torch.exp(-self.gamma * dt)
        sigma = torch.sqrt((self.kT / self.mass) * (1 - alpha**2))
        v = alpha * v + sigma * torch.randn_like(v)
        x = x + (dt / 2) * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        return x, v

    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float | torch.Tensor, n_steps: int, store_every: int = 1, final_only: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities) each (n_stored, ...).
        
        Args:
            x0: Initial positions (..., dim)
            v0: Initial velocities (..., dim), or None to sample from thermal distribution
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps (ignored if final_only=True)
            final_only: If True, only return final state with shape (1, ..., dim)
            
        Returns:
            (traj_x, traj_v): Trajectories of shape (n_stored, ..., dim) or (1, ..., dim) if final_only
        """
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        
        state = (x0, v)
        step_fn = lambda x, v: self.step(x, v, force_fn, dt)
        
        return self._integrate(state, step_fn, n_steps, store_every, final_only)


class VelocityVerlet(Integrator):
    """Symplectic velocity Verlet integrator (NVE ensemble).
    
    Preserves phase-space volume and has excellent energy conservation.
    No thermostat - samples microcanonical ensemble.
    
    Args:
        mass: Particle mass. Differentiable parameter.
    """
    
    def __init__(self, mass: float = 1.0):
        super().__init__()
        self.mass = nn.Parameter(torch.tensor(mass))
    
    def step(self, x: torch.Tensor, v: torch.Tensor, force_fn: ForceFunc,
             dt: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single Verlet step. Returns (new_x, new_v)."""
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        return x, v
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor, force_fn: ForceFunc,
            dt: float | torch.Tensor, n_steps: int, store_every: int = 1, final_only: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities).
        
        Args:
            x0: Initial positions (..., dim)
            v0: Initial velocities (..., dim)
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps (ignored if final_only=True)
            final_only: If True, only return final state with shape (1, ..., dim)
            
        Returns:
            (traj_x, traj_v): Trajectories of shape (n_stored, ..., dim) or (1, ..., dim) if final_only
        """
        state = (x0, v0)
        step_fn = lambda x, v: self.step(x, v, force_fn, dt)
        
        return self._integrate(state, step_fn, n_steps, store_every, final_only)


class NoseHoover(Integrator):
    """Single Nosé-Hoover thermostat for deterministic canonical sampling.
    
    Uses the Kleinerman 08 symmetric integration scheme. Simpler and more
    robust than chain thermostats for many applications.
    
    Reference: Kleinerman et al., J. Chem. Phys. 128, 124109 (2008)
    
    Args:
        kT: Thermal energy (target temperature). Differentiable parameter.
        mass: Particle mass. Differentiable parameter.
        Q: Thermostat mass (coupling strength). Larger Q = slower coupling.
    """
    
    def __init__(self, kT: float = 1.0, mass: float = 1.0, Q: float = 1.0):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(kT))
        self.mass = nn.Parameter(torch.tensor(mass))
        self.Q = nn.Parameter(torch.tensor(Q))
    
    def step(self, x: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
             force_fn: ForceFunc, dt: float | torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single Nosé-Hoover step using Kleinerman 08 scheme.
        
        Args:
            x: positions (..., dim)
            v: velocities (..., dim)  
            alpha: thermostat variable (...,)
            force_fn: force function
            dt: time step
            
        Returns:
            (new_x, new_v, new_alpha)
        """
        # Number of degrees of freedom (per batch element)
        ndof = x.shape[-1]
        
        # Compute v^2 (kinetic energy * 2 / mass)
        v2 = (v**2).sum(dim=-1)
        
        # First thermostat half-step
        # dα/dt = (v² - ndof·kT) / Q (Nosé-Hoover equation)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / self.Q
        v = v * torch.exp(-alpha.unsqueeze(-1) * dt / 2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / self.Q
        
        # Velocity-Verlet for physical degrees of freedom
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        
        # Second thermostat half-step
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / self.Q
        v = v * torch.exp(-alpha.unsqueeze(-1) * dt / 2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / self.Q
        
        return x, v, alpha

    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float | torch.Tensor, n_steps: int, store_every: int = 1, final_only: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities).
        
        Args:
            x0: Initial positions (..., dim)
            v0: Initial velocities (..., dim), or None to sample from thermal distribution
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps (ignored if final_only=True)
            final_only: If True, only return final state with shape (1, ..., dim) for memory efficiency
            
        Returns:
            (traj_x, traj_v): Trajectories of shape (n_stored, ..., dim) or (1, ..., dim) if final_only
        """
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        alpha = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        
        state = (x0, v, alpha)
        step_fn = lambda x, v, alpha: self.step(x, v, alpha, force_fn, dt)
        
        # We only return x and v, so store_indices=[0, 1]
        return self._integrate(state, step_fn, n_steps, store_every, final_only, store_indices=[0, 1])


class NoseHooverChain(Integrator):
    """Nosé-Hoover chain thermostat for deterministic canonical sampling.
    
    Extends phase space with thermostat variables to sample NVT ensemble
    without stochastic noise. Chain length > 1 improves ergodicity.
    
    Note: For simple systems, NoseHoover (single thermostat) often works better.
    
    Args:
        kT: Thermal energy. Differentiable parameter.
        mass: Particle mass. Differentiable parameter.
        Q: Thermostat mass. Differentiable parameter.
        n_chain: Number of thermostat variables (not differentiable, structural).
    """
    
    def __init__(self, kT: float = 1.0, mass: float = 1.0, 
                 Q: float = 1.0, n_chain: int = 2):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(kT))
        self.mass = nn.Parameter(torch.tensor(mass))
        self.Q = nn.Parameter(torch.tensor(Q))
        self.n_chain = n_chain  # structural parameter, not differentiable
    
    def step(self, x: torch.Tensor, v: torch.Tensor, xi: torch.Tensor,
             force_fn: ForceFunc, dt: float | torch.Tensor, ndof: int
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single NHC step. Returns (new_x, new_v, new_xi).
        
        Uses functional updates to maintain differentiability.
        """
        KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
        G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
        
        # Store xi values in list for functional updates
        xi_list = [xi[..., j] for j in range(self.n_chain)]
        
        # First thermostat half-step (chain from end to start)
        for j in range(self.n_chain - 1, -1, -1):
            if j == self.n_chain - 1:
                xi_list[j] = xi_list[j] + (dt / 4) * G
            else:
                xi_list[j] = xi_list[j] * torch.exp(-xi_list[j+1] * dt / 8)
                xi_list[j] = xi_list[j] + (dt / 4) * G
                xi_list[j] = xi_list[j] * torch.exp(-xi_list[j+1] * dt / 8)
            if j == 0:
                v = v * torch.exp(-xi_list[0].unsqueeze(-1) * dt / 2)
                KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
            G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q if j == 0 else \
                (self.Q * xi_list[j-1]**2 - self.kT) / self.Q
        
        # Velocity-Verlet for physical degrees of freedom
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        
        # Second thermostat half-step (chain from start to end)
        KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
        G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
        
        for j in range(self.n_chain):
            if j == 0:
                v = v * torch.exp(-xi_list[0].unsqueeze(-1) * dt / 2)
                KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
                G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
            if j == self.n_chain - 1:
                xi_list[j] = xi_list[j] + (dt / 4) * G
            else:
                xi_list[j] = xi_list[j] * torch.exp(-xi_list[j+1] * dt / 8)
                xi_list[j] = xi_list[j] + (dt / 4) * G
                xi_list[j] = xi_list[j] * torch.exp(-xi_list[j+1] * dt / 8)
            G = (self.Q * xi_list[j]**2 - self.kT) / self.Q
        
        # Stack back to tensor
        xi_new = torch.stack(xi_list, dim=-1)
        return x, v, xi_new
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float | torch.Tensor, n_steps: int, store_every: int = 1, final_only: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities).
        
        Args:
            x0: Initial positions (..., dim)
            v0: Initial velocities (..., dim), or None to sample from thermal distribution
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps (ignored if final_only=True)
            final_only: If True, only return final state with shape (1, ..., dim)
            
        Returns:
            (traj_x, traj_v): Trajectories of shape (n_stored, ..., dim) or (1, ..., dim) if final_only
        """
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        ndof = x0.shape[-1]
        xi_shape = x0.shape[:-1] + (self.n_chain,)
        xi = torch.zeros(xi_shape, device=x0.device, dtype=x0.dtype)
        
        state = (x0, v, xi)
        step_fn = lambda x, v, xi: self.step(x, v, xi, force_fn, dt, ndof)
        
        return self._integrate(state, step_fn, n_steps, store_every, final_only, store_indices=[0, 1])


class ESH(Integrator):
    """Energy Sampling Hamiltonian dynamics for deterministic ergodic sampling.
    
    Uses non-Newtonian kinetic energy K(v) = d/2 * log(v²/d).
    The time spent in region dx is proportional to exp(-E(x)), giving exact
    Boltzmann sampling without stochastic noise.
    
    Reference: Ver Steeg & Galstyan, "Hamiltonian Dynamics with Non-Newtonian 
    Momentum for Rapid Sampling", NeurIPS 2021. arXiv:2111.02434
    
    Key properties:
    - Deterministic (excellent for gradient computation)
    - Ergodic sampling of target distribution  
    - Fast convergence compared to MCMC
    - Symplectic (volume-preserving)
    
    Note: Works best in d >= 2 dimensions. Uses scaled dynamics with
    unit velocity u = v/|v| and log-magnitude r = log|v|.
    
    Stability: The integrator requires eps * |grad| / d << 1 for numerical
    stability. Use max_grad_norm to clip gradients in high-curvature regions.
    
    Args:
        eps: Default step size. Differentiable parameter.
        max_grad_norm: Maximum gradient norm for stability. If None, no clipping.
            Recommended: set to d / eps for stability (default: 10.0).
    """
    
    def __init__(self, eps: float = 0.1, max_grad_norm: float | None = 10.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.max_grad_norm = max_grad_norm
    
    def _u_r_step(self, u: torch.Tensor, r: torch.Tensor, 
                  grad: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Update unit velocity u and log-magnitude r. Vectorized over batch."""
        # Dimension d from last axis
        d = u.shape[-1]
        
        # Handle potential infinite gradients (e.g. from exploding potentials)
        # Replace Inf with large finite number to preserve direction for clipping
        if not torch.isfinite(grad).all():
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1e30, neginf=-1e30)
            
        # Gradient norm and unit vector
        g_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        
        # Clip gradient norm for numerical stability
        # The integrator requires eps * |grad| / d << 1
        if self.max_grad_norm is not None:
            g_norm = g_norm.clamp(max=self.max_grad_norm)
            grad = grad * (g_norm / grad.norm(dim=-1, keepdim=True).clamp(min=1e-10))
        
        grad_e = grad / g_norm  # Unit vector in gradient direction
        
        # u · (-e) where e = grad/|grad|
        u_dot_e = -(u * grad_e).sum(dim=-1, keepdim=True)
        
        # Coefficients for the update (Eq. from paper appendix)
        exp_term = torch.exp(-2 * eps * g_norm / d)
        A2 = (u_dot_e - 1.0) * exp_term
        A = 1.0 + u_dot_e + A2
        B = 2.0 * torch.exp(-eps * g_norm / d)
        
        # Perpendicular component
        perp = u + grad_e * u_dot_e
        
        # Update u: handle u·e = -1 edge case
        u_new = torch.where(
            u_dot_e > -0.999,
            A * (-grad_e) + B * perp,
            grad_e  # When u·e ≈ -1, flip direction
        )
        # Normalize to unit vector
        u_new = u_new / u_new.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        
        # Update r (log velocity magnitude)
        Z = 1.0 + u_dot_e - A2
        delta_r = torch.where(
            u_dot_e.squeeze(-1) > -0.999,
            (eps * g_norm.squeeze(-1) / d + torch.log(0.5 * Z.squeeze(-1).clamp(min=1e-10))),
            -eps * g_norm.squeeze(-1) / d
        )
        r_new = r + delta_r
        
        return u_new, r_new
    
    def step(self, x: torch.Tensor, u: torch.Tensor, r: torch.Tensor,
             grad_fn: GradFunc, dt: float | torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single ESH leapfrog step with scaled dynamics.
        
        Args:
            x: positions (..., dim)
            u: unit velocity (..., dim), |u| = 1
            r: log velocity magnitude (...,)
            grad_fn: returns ∇E(x)
            dt: step size (uses self.eps if None)
        
        Returns:
            (new_x, new_u, new_r)
        """
        eps = dt if dt is not None else self.eps
        
        # Half step in u, r
        grad = grad_fn(x)
        u, r = self._u_r_step(u, r, grad, eps / 2)
        
        # Full step in x (move along unit velocity direction)
        x = x + eps * u
        
        # Half step in u, r
        grad = grad_fn(x)
        u, r = self._u_r_step(u, r, grad, eps / 2)
        
        return x, u, r
    
    def run(self, x0: torch.Tensor, u0: torch.Tensor | None, grad_fn: GradFunc,
            n_steps: int, dt: float | torch.Tensor | None = None, store_every: int = 1,
            final_only: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ESH trajectory.
        
        Args:
            x0: initial positions (..., dim)
            u0: initial unit velocities (if None, random unit vectors)
            grad_fn: gradient of energy ∇E(x)
            n_steps: number of integration steps
            dt: step size (uses self.eps if None)
            store_every: store every N steps (ignored if final_only=True)
            final_only: If True, only return final state with shape (1, ..., dim)
            
        Returns:
            (positions, unit_velocities, log_v_magnitudes) each of shape (n_stored, ...) or (1, ...) if final_only
        """
        eps = dt if dt is not None else self.eps
        x = x0
        
        if u0 is None:
            u = torch.randn_like(x0)
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        else:
            u = u0
        
        r = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        
        state = (x, u, r)
        step_fn = lambda x, u, r: self.step(x, u, r, grad_fn, dt)
        
        return self._integrate(state, step_fn, n_steps, store_every, final_only)


class GLE(Integrator):
    """Generalized Langevin Equation with colored noise.
    
    Implements dynamics with memory kernel using Prony series decomposition:
        m dv/dt = F(x) - Σᵢ sᵢ + η(t)
    
    where sᵢ are auxiliary variables satisfying:
        dsᵢ = -γᵢ sᵢ dt + cᵢ v dt + √(2 cᵢ kT γᵢ) dW
    
    This corresponds to memory kernel K(t) = Σᵢ cᵢ exp(-γᵢ t).
    
    Properties:
    - Colored noise for better sampling of multi-scale systems
    - Memory effects can accelerate barrier crossing
    - Reduces to standard Langevin when n_modes=1 and c=γ
    
    Reference: Ceriotti et al., J. Chem. Theory Comput. 6, 1170 (2010)
    
    Args:
        kT: Temperature. Differentiable parameter.
        mass: Particle mass. Differentiable parameter.
        gamma: Decay rates for memory kernel modes. Differentiable parameter.
        c: Coupling strengths for each mode. Differentiable parameter.
    """
    
    def __init__(self, kT: float = 1.0, mass: float = 1.0,
                 gamma: list[float] | None = None,
                 c: list[float] | None = None):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(kT))
        self.mass = nn.Parameter(torch.tensor(mass))
        
        # Default: single mode (standard Langevin-like)
        if gamma is None:
            gamma = [1.0]
        if c is None:
            c = list(gamma)
        
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.c = nn.Parameter(torch.tensor(c))
        self.n_modes = len(gamma)
    
    def step(self, x: torch.Tensor, v: torch.Tensor, s: torch.Tensor,
             force_fn: ForceFunc, dt: float | torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single GLE step using BAOAB-like splitting.
        
        Args:
            x: positions (..., dim)
            v: velocities (..., dim)
            s: auxiliary momenta (..., dim, n_modes)
            force_fn: force function F(x) = -∇U(x)
            dt: time step
            
        Returns:
            (new_x, new_v, new_s)
        """
        # Ensure parameters are on the same device as input
        gamma = self.gamma.to(x.device)
        c = self.c.to(x.device)
        kT = self.kT.to(x.device)
        mass = self.mass.to(x.device)
        
        # Friction from auxiliary variables
        friction = s.sum(dim=-1)
        
        # B: Half-step velocity
        v = v + (dt / 2) * (force_fn(x) - friction) / mass
        
        # A: Half-step position
        x = x + (dt / 2) * v
        
        # O: Update auxiliary variables (vectorized)
        alpha = torch.exp(-gamma * dt)
        # Noise variance from fluctuation-dissipation
        sigma = torch.sqrt(c * kT * (1 - alpha**2))
        noise = torch.randn_like(s) * sigma
        
        # Drift coefficient
        drift_coef = (c / gamma) * (1 - alpha)
        s = alpha * s + drift_coef * v.unsqueeze(-1) + noise
        
        # A: Half-step position
        x = x + (dt / 2) * v
        
        # B: Half-step velocity with updated friction
        friction = s.sum(dim=-1)
        v = v + (dt / 2) * (force_fn(x) - friction) / mass
        
        return x, v, s
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float | torch.Tensor, n_steps: int, store_every: int = 1, final_only: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GLE trajectory. Returns (positions, velocities).
        
        Args:
            x0: Initial positions (..., dim)
            v0: Initial velocities (..., dim), or None to sample from thermal distribution
            force_fn: Force function
            dt: Time step
            n_steps: Number of integration steps
            store_every: Store trajectory every N steps (ignored if final_only=True)
            final_only: If True, only return final state with shape (1, ..., dim)
            
        Returns:
            (traj_x, traj_v): Trajectories of shape (n_stored, ..., dim) or (1, ..., dim) if final_only
        """
        device, dtype = x0.device, x0.dtype
        x = x0
        
        # Ensure parameters are on correct device
        kT = self.kT.to(device)
        mass = self.mass.to(device)
        c = self.c.to(device)
        
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(kT / mass)
        
        # Initialize auxiliary variables at thermal equilibrium
        s_shape = x0.shape + (self.n_modes,)
        s = torch.randn(s_shape, device=device, dtype=dtype) * torch.sqrt(c * kT)
        
        state = (x, v, s)
        step_fn = lambda x, v, s: self.step(x, v, s, force_fn, dt)
        
        return self._integrate(state, step_fn, n_steps, store_every, final_only, store_indices=[0, 1])


def kinetic_energy(v: torch.Tensor, mass: float = 1.0) -> torch.Tensor:
    """Compute kinetic energy. v: (..., dim) -> (...)."""
    return 0.5 * mass * (v**2).sum(dim=-1)


def temperature(v: torch.Tensor, mass: float = 1.0) -> torch.Tensor:
    """Compute instantaneous temperature from velocities."""
    ndof = v.shape[-1]
    KE = kinetic_energy(v, mass)
    return 2 * KE / ndof


if __name__ == "__main__":
    """Quick sanity check for integrators.
    
    For full demos and benchmarks, see scripts/demo_*.py and scripts/benchmark_*.py
    """
    print("Integrators sanity check...")
    
    # Test each integrator runs without error
    x0 = torch.randn(10, 2)
    v0 = torch.randn(10, 2)
    force_fn = lambda x: -x  # Harmonic
    grad_fn = lambda x: x
    dt = 0.01
    n_steps = 100
    
    # OverdampedLangevin
    od = OverdampedLangevin(gamma=1.0, kT=1.0)
    traj = od.run(x0, force_fn, dt, n_steps)
    assert traj.shape == (n_steps // 1 + 1, 10, 2), f"OverdampedLangevin: {traj.shape}"
    print(f"  OverdampedLangevin: OK, final x mean = {traj[-1].mean():.3f}")
    
    # BAOAB
    baoab = BAOAB(gamma=1.0, kT=1.0, mass=1.0)
    traj_x, traj_v = baoab.run(x0, v0, force_fn, dt, n_steps)
    assert traj_x.shape[0] == n_steps + 1
    print(f"  BAOAB: OK, final x mean = {traj_x[-1].mean():.3f}")
    
    # VelocityVerlet
    vv = VelocityVerlet(mass=1.0)
    traj_x, traj_v = vv.run(x0, v0, force_fn, dt, n_steps)
    assert traj_x.shape[0] == n_steps + 1
    print(f"  VelocityVerlet: OK, final x mean = {traj_x[-1].mean():.3f}")
    
    # NoseHoover
    nh = NoseHoover(kT=1.0, mass=1.0, Q=1.0)
    traj_x, traj_v = nh.run(x0, v0, force_fn, dt, n_steps)
    assert traj_x.shape[0] == n_steps + 1
    print(f"  NoseHoover: OK, final x mean = {traj_x[-1].mean():.3f}")
    
    # NoseHooverChain
    nhc = NoseHooverChain(kT=1.0, mass=1.0, Q=1.0, n_chain=2)
    traj_x, traj_v = nhc.run(x0, v0, force_fn, dt, n_steps)
    assert traj_x.shape[0] == n_steps + 1
    print(f"  NoseHooverChain: OK, final x mean = {traj_x[-1].mean():.3f}")
    
    # ESH
    esh = ESH(eps=0.1)
    u0 = torch.randn(10, 2)
    u0 = u0 / u0.norm(dim=-1, keepdim=True)
    traj_x, traj_u, traj_r = esh.run(x0, u0, grad_fn, n_steps=n_steps)
    assert traj_x.shape[0] == n_steps + 1
    print(f"  ESH: OK, final x mean = {traj_x[-1].mean():.3f}")
    
    # GLE
    gle = GLE(kT=1.0, mass=1.0, gamma=[1.0, 2.0], c=[1.0, 2.0])
    traj_x, traj_v = gle.run(x0, v0, force_fn, dt, n_steps)
    assert traj_x.shape[0] == n_steps + 1
    print(f"  GLE: OK, final x mean = {traj_x[-1].mean():.3f}")
    
    # Test gradient flow
    x0_grad = torch.randn(5, 2, requires_grad=True)
    traj = od.run(x0_grad, force_fn, dt, 10)
    loss = traj[-1].sum()
    loss.backward()
    assert x0_grad.grad is not None, "Gradient should flow through"
    print(f"  Gradient flow: OK")
    
    print("\nAll integrators passed sanity check!")
