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


class OverdampedLangevin(nn.Module):
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
    
    def step(self, x: torch.Tensor, force_fn: ForceFunc, dt: float) -> torch.Tensor:
        """Single integration step. Returns new positions."""
        force = force_fn(x)
        noise_scale = torch.sqrt(2 * self.kT * dt / self.gamma)
        return x + (force / self.gamma) * dt + noise_scale * torch.randn_like(x)
    
    def run(self, x0: torch.Tensor, force_fn: ForceFunc, dt: float, 
            n_steps: int, store_every: int = 1) -> torch.Tensor:
        """Run trajectory. Returns (n_stored, ..., dim) positions."""
        x = x0
        n_stored = n_steps // store_every + 1
        traj = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj[0] = x0
        
        idx = 1
        for i in range(1, n_steps + 1):
            x = self.step(x, force_fn, dt)
            if i % store_every == 0:
                traj[idx] = x
                idx += 1
        return traj


class BAOAB(nn.Module):
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
             dt: float) -> tuple[torch.Tensor, torch.Tensor]:
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
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities) each (n_stored, ...)."""
        x = x0
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj_v = torch.empty((n_stored, *v.shape), dtype=v.dtype, device=v.device)
        
        traj_x[0] = x0
        traj_v[0] = v
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, v = self.step(x, v, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx] = x
                traj_v[idx] = v
                idx += 1
        return traj_x, traj_v


class VelocityVerlet(nn.Module):
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
             dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Single Verlet step. Returns (new_x, new_v)."""
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        return x, v
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor, force_fn: ForceFunc,
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities)."""
        x, v = x0, v0
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj_v = torch.empty((n_stored, *v0.shape), dtype=v0.dtype, device=v0.device)
        
        traj_x[0] = x0
        traj_v[0] = v0
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, v = self.step(x, v, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx] = x
                traj_v[idx] = v
                idx += 1
        return traj_x, traj_v


class NoseHoover(nn.Module):
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
             force_fn: ForceFunc, dt: float
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
        alpha = alpha + (dt / 4) * (v2 / self.kT - ndof)
        v = v * torch.exp(-alpha.unsqueeze(-1) * dt / 2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 / self.kT - ndof)
        
        # Velocity-Verlet for physical degrees of freedom
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        
        # Second thermostat half-step
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 / self.kT - ndof)
        v = v * torch.exp(-alpha.unsqueeze(-1) * dt / 2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 / self.kT - ndof)
        
        return x, v, alpha
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities)."""
        x = x0
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        alpha = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj_v = torch.empty((n_stored, *v.shape), dtype=v.dtype, device=v.device)
        
        traj_x[0] = x0
        traj_v[0] = v
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, v, alpha = self.step(x, v, alpha, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx] = x
                traj_v[idx] = v
                idx += 1
        return traj_x, traj_v


class NoseHooverChain(nn.Module):
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
             force_fn: ForceFunc, dt: float, ndof: int
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
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities)."""
        x = x0
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        ndof = x0.shape[-1]
        xi_shape = x0.shape[:-1] + (self.n_chain,)
        xi = torch.zeros(xi_shape, device=x0.device, dtype=x0.dtype)
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj_v = torch.empty((n_stored, *v.shape), dtype=v.dtype, device=v.device)
        
        traj_x[0] = x0
        traj_v[0] = v
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, v, xi = self.step(x, v, xi, force_fn, dt, ndof)
            if i % store_every == 0:
                traj_x[idx] = x
                traj_v[idx] = v
                idx += 1
        return traj_x, traj_v


class ESH(nn.Module):
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
             grad_fn: GradFunc, dt: float | None = None
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
            n_steps: int, dt: float | None = None, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ESH trajectory.
        
        Args:
            x0: initial positions (..., dim)
            u0: initial unit velocities (if None, random unit vectors)
            grad_fn: gradient of energy ∇E(x)
            n_steps: number of integration steps
            dt: step size (uses self.eps if None)
            store_every: store every N steps
            
        Returns:
            (positions, unit_velocities, log_v_magnitudes)
        """
        eps = dt if dt is not None else self.eps
        x = x0
        
        if u0 is None:
            u = torch.randn_like(x0)
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        else:
            u = u0
        
        r = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=x0.dtype, device=x0.device)
        traj_u = torch.empty((n_stored, *u.shape), dtype=u.dtype, device=u.device)
        traj_r = torch.empty((n_stored, *r.shape), dtype=r.dtype, device=r.device)
        
        traj_x[0] = x0
        traj_u[0] = u
        traj_r[0] = r
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, u, r = self.step(x, u, r, grad_fn, eps)
            if i % store_every == 0:
                traj_x[idx] = x
                traj_u[idx] = u
                traj_r[idx] = r
                idx += 1
        
        return traj_x, traj_u, traj_r


class GLE(nn.Module):
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
             force_fn: ForceFunc, dt: float
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
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GLE trajectory. Returns (positions, velocities)."""
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
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored, *x0.shape), dtype=dtype, device=device)
        traj_v = torch.empty((n_stored, *v.shape), dtype=dtype, device=device)
        
        traj_x[0] = x0
        traj_v[0] = v
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, v, s = self.step(x, v, s, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx] = x
                traj_v[idx] = v
                idx += 1
        
        return traj_x, traj_v


def kinetic_energy(v: torch.Tensor, mass: float = 1.0) -> torch.Tensor:
    """Compute kinetic energy. v: (..., dim) -> (...)."""
    return 0.5 * mass * (v**2).sum(dim=-1)


def temperature(v: torch.Tensor, mass: float = 1.0) -> torch.Tensor:
    """Compute instantaneous temperature from velocities."""
    ndof = v.shape[-1]
    KE = kinetic_energy(v, mass)
    return 2 * KE / ndof


if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from .potentials import DoubleWell, DoubleWell2D, Harmonic
    
    # Plotting style
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
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    
    # Common colormap and style
    # Distinct colors for better separation
    colors = {
        'Overdamped': '#1f77b4',  # Blue
        'BAOAB': '#ff7f0e',       # Orange
        'GLE': '#2ca02c',         # Green
        'NH': '#d62728',          # Red
        'ESH': '#9467bd',         # Purple
        'VelocityVerlet': '#8c564b', # Brown
        'NoseHooverChain': '#e377c2' # Pink
    }
    
    # Setup for 1D double well
    dw = DoubleWell()
    kT = 0.5
    dt = 0.01
    n_steps = 50000
    n_batch = 50
    
    def force_fn_1d(x):
        return dw.force(x.unsqueeze(-1)).squeeze(-1)
        
    def add_1d_density_inset(ax, samples, potential, kT, color, label=''):
        # Inset for 1D density
        ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
        
        # Empirical density (histogram)
        ax_ins.hist(samples, bins=40, density=True, range=(-2.5, 2.5),
                   color=color, alpha=0.6, edgecolor='none')
        
        # Theoretical density
        x_th = torch.linspace(-2.5, 2.5, 200)
        u_th = potential.energy(x_th)
        p_th = torch.exp(-u_th / kT)
        p_th = p_th / (p_th.sum() * (x_th[1] - x_th[0]))
        ax_ins.plot(x_th.numpy(), p_th.detach().numpy(), 'k-', lw=1.5, alpha=0.8)
        
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        ax_ins.set_facecolor('none')
        for spine in ax_ins.spines.values():
            spine.set_visible(False)
    
    # 1. Overdamped Langevin
    ax = axes[0, 0]
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    x0 = torch.full((n_batch,), -1.0)
    traj_od = integrator.run(x0, force_fn_1d, dt, n_steps, store_every=10)
    t = np.arange(traj_od.shape[0]) * dt * 10
    for i in range(min(3, n_batch)):
        ax.plot(t, traj_od[:, i].detach().numpy(), alpha=0.75, lw=1.8, color=colors['Overdamped'])
    ax.axhline(1, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.axhline(-1, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('Overdamped Langevin', fontweight='bold')
    ax.set_axisbelow(True)
    # Add density inset
    burn_in = 2000
    samples_od = traj_od[burn_in//10:].flatten().detach().numpy()
    add_1d_density_inset(ax, samples_od, dw, kT, colors['Overdamped'])
    
    # 2. BAOAB
    ax = axes[0, 1]
    integrator = BAOAB(gamma=1.0, kT=kT, mass=1.0)
    traj_baoab, _ = integrator.run(x0, None, force_fn_1d, dt, n_steps, store_every=10)
    for i in range(min(3, n_batch)):
        ax.plot(t, traj_baoab[:, i].detach().numpy(), alpha=0.75, lw=1.8, color=colors['BAOAB'])
    ax.axhline(1, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.axhline(-1, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('BAOAB', fontweight='bold')
    ax.set_axisbelow(True)
    # Add density inset
    samples_baoab = traj_baoab[burn_in//10:].flatten().detach().numpy()
    add_1d_density_inset(ax, samples_baoab, dw, kT, colors['BAOAB'])
    
    # 3. GLE (colored noise)
    ax = axes[0, 2]
    gle = GLE(kT=kT, mass=1.0, gamma=[0.5, 2.0], c=[0.3, 1.0])
    traj_gle, _ = gle.run(x0, None, force_fn_1d, dt, n_steps, store_every=10)
    for i in range(min(3, n_batch)):
        ax.plot(t, traj_gle[:, i].detach().numpy(), alpha=0.75, lw=1.8, color=colors['GLE'])
    ax.axhline(1, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.axhline(-1, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('GLE (colored noise)', fontweight='bold')
    ax.set_axisbelow(True)
    # Add density inset
    samples_gle = traj_gle[burn_in//10:].flatten().detach().numpy()
    add_1d_density_inset(ax, samples_gle, dw, kT, colors['GLE'])
    
    # 4. 2D Double Well Sampling (Row 2)
    # Use DoubleWell2D instead of Harmonic
    dw2d = DoubleWell2D(barrier_height=1.0, k_y=1.0)
    kT_2d = 0.5 # Lower temperature to see hopping
    
    def grad_dw2d(x):
        return -dw2d.force(x)
    
    # Prepare background contours
    x_grid = torch.linspace(-2.0, 2.0, 100)
    y_grid = torch.linspace(-2.0, 2.0, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    xy_grid = torch.stack([X, Y], dim=-1)
    U_grid = dw2d.energy(xy_grid)
    
    # Helper for 2D KDE insets
    def add_2d_kde_inset(ax, samples, weights=None, color_map='Blues'):
        ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
        
        # Kernel Density Estimation
        try:
            # Subsample for KDE if too large
            if len(samples) > 5000:
                idx = np.random.choice(len(samples), 5000, p=weights if weights is not None else None, replace=False)
                kde_samples = samples[idx]
                kde_weights = weights[idx] if weights is not None else None
            else:
                kde_samples = samples
                kde_weights = weights
                
            x = kde_samples[:, 0]
            y = kde_samples[:, 1]
            
            # Create grid for KDE evaluation
            xmin, xmax = -2.0, 2.0
            ymin, ymax = -2.0, 2.0
            xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])
            
            kernel = gaussian_kde(values, weights=kde_weights)
            f = np.reshape(kernel(positions).T, xx.shape)
            
            ax_ins.contourf(xx, yy, f, cmap=color_map, levels=10)
            ax_ins.set_xlim(xmin, xmax)
            ax_ins.set_ylim(ymin, ymax)
        except Exception:
            # Fallback if KDE fails (e.g. singular matrix)
            pass
            
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        ax_ins.set_facecolor('white')
        for spine in ax_ins.spines.values():
            spine.set_visible(True) # Keep box for 2D density
    
    # Run samplers on 2D Double Well
    n_steps_2d = 50000
    
    # ESH
    torch.manual_seed(42)
    esh = ESH(eps=0.1)
    x0_esh = torch.randn(20, 2)
    u0_esh = torch.randn(20, 2)
    u0_esh = u0_esh / u0_esh.norm(dim=-1, keepdim=True)
    
    traj_esh_2d, _, traj_r = esh.run(x0_esh, u0_esh, grad_dw2d, n_steps=n_steps_2d, dt=0.1, store_every=1)
    
    burn_in = 5000
    esh_x = traj_esh_2d[burn_in:].detach().numpy()
    esh_r = traj_r[burn_in:].detach().numpy()
    esh_weights = np.exp(esh_r)
    esh_weights = esh_weights / esh_weights.sum()
    esh_samples = esh_x.reshape(-1, 2)
    esh_w_flat = esh_weights.flatten()
    
    # Nosé-Hoover
    nh_2d = NoseHoover(kT=kT_2d, mass=1.0, Q=1.0)
    x0_nh = torch.randn(20, 2)
    traj_nh_2d, _ = nh_2d.run(x0_nh, None, dw2d.force, dt=0.05, n_steps=n_steps_2d, store_every=1)
    nh_samples = traj_nh_2d[5000:].reshape(-1, 2).detach().numpy()
    
    # BAOAB
    baoab_2d = BAOAB(gamma=1.0, kT=kT_2d, mass=1.0)
    x0_baoab = torch.randn(200, 2) # More chains for baoab to cover space
    traj_baoab_2d, _ = baoab_2d.run(x0_baoab, None, dw2d.force, dt=0.05, n_steps=10000, store_every=10)
    baoab_samples = traj_baoab_2d[100:].reshape(-1, 2).detach().numpy()
    
    # Plotting 2D
    # ESH
    ax = axes[1, 0]
    ax.contour(X.numpy(), Y.numpy(), U_grid.detach().numpy(), levels=np.linspace(0, 5, 10), colors='k', alpha=0.2)
    
    # Importance resampling for scatter
    esh_idx = np.random.choice(len(esh_samples), size=5000, p=esh_w_flat)
    ax.scatter(esh_samples[esh_idx, 0], esh_samples[esh_idx, 1], s=5, alpha=0.3, 
               c=colors['ESH'], edgecolors='none')
               
    ax.set_title('ESH (2D Double Well)', fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    add_2d_kde_inset(ax, esh_samples, esh_w_flat, color_map='Purples')
    
    # NH
    ax = axes[1, 1]
    ax.contour(X.numpy(), Y.numpy(), U_grid.detach().numpy(), levels=np.linspace(0, 5, 10), colors='k', alpha=0.2)
    ax.scatter(nh_samples[::10, 0], nh_samples[::10, 1], s=5, alpha=0.3,
               c=colors['NH'], edgecolors='none')
    ax.set_title('Nosé-Hoover (2D Double Well)', fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    add_2d_kde_inset(ax, nh_samples, None, color_map='Reds')
    
    # BAOAB
    ax = axes[1, 2]
    ax.contour(X.numpy(), Y.numpy(), U_grid.detach().numpy(), levels=np.linspace(0, 5, 10), colors='k', alpha=0.2)
    ax.scatter(baoab_samples[::2, 0], baoab_samples[::2, 1], s=5, alpha=0.3,
               c=colors['BAOAB'], edgecolors='none')
    ax.set_title('BAOAB (2D Double Well)', fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    add_2d_kde_inset(ax, baoab_samples, None, color_map='Oranges')
    
    # 7. Benchmark Plot (Forward vs Backward)
    ax = axes[2, 0]
    # Spanning all columns in the bottom row
    ax.remove()
    ax = axes[2, 1]
    ax.remove()
    ax = axes[2, 2]
    ax.remove()
    
    # Create a new axis spanning the bottom row
    gs = axes[0, 0].get_gridspec()
    ax_bench = fig.add_subplot(gs[2, :])
    
    def run_benchmark_for_plot():
        # Setup for benchmark
        dim = 64
        n_particles = 256
        n_steps = 100
        dt = 0.01
        device = torch.device('cpu')
        
        integrators_list = [
            ("Overdamped", OverdampedLangevin(gamma=1.0, kT=1.0)),
            ("BAOAB", BAOAB(gamma=1.0, kT=1.0, mass=1.0)),
            ("Verlet", VelocityVerlet(mass=1.0)),
            ("NH", NoseHoover(kT=1.0, mass=1.0, Q=1.0)),
            ("NHC", NoseHooverChain(kT=1.0, mass=1.0, Q=1.0, n_chain=2)),
            ("ESH", ESH(eps=0.1)),
            ("GLE", GLE(kT=1.0, mass=1.0, gamma=[1.0, 2.0], c=[1.0, 2.0]))
        ]
        
        fwd_times = []
        bwd_times = []
        names = []
        
        # Simple Harmonic force
        def force_fn(x):
            return -x
        def grad_fn(x):
            return x
            
        x0 = torch.randn(n_particles, dim, device=device, requires_grad=True)
        v0 = torch.randn(n_particles, dim, device=device, requires_grad=True)
        
        import time
        
        for name, integrator in integrators_list:
            integrator = integrator.to(device)
            names.append(name)
            
            # Helper to run integrator
            def run_int():
                if name == "ESH":
                    return integrator.run(x0, None, grad_fn, n_steps=n_steps)
                elif name == "Overdamped":
                    return integrator.run(x0, force_fn, dt=dt, n_steps=n_steps)
                elif name == "Verlet":
                    return integrator.run(x0, v0, force_fn, dt=dt, n_steps=n_steps)
                else:
                    return integrator.run(x0, None, force_fn, dt=dt, n_steps=n_steps)

            # Warmup
            try:
                run_int()
            except Exception:
                pass
                
            # Forward
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            out = run_int()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_fwd = time.perf_counter() - t0
            fwd_times.append(t_fwd)
            
            # Backward
            if isinstance(out, tuple):
                loss = out[0].sum()
            else:
                loss = out.sum()
            
            # Reset gradients
            if x0.grad is not None: x0.grad.zero_()
            if v0.grad is not None: v0.grad.zero_()
            for p in integrator.parameters():
                if p.grad is not None: p.grad.zero_()
                
            t0 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_bwd = time.perf_counter() - t0
            bwd_times.append(t_bwd)
            
        return names, fwd_times, bwd_times

    names, fwd_times, bwd_times = run_benchmark_for_plot()
    
    # Bar plot
    x = np.arange(len(names))
    width = 0.35
    
    rects1 = ax_bench.bar(x - width/2, fwd_times, width, label='Forward (Execution)', color='#1f77b4', alpha=0.8)
    rects2 = ax_bench.bar(x + width/2, bwd_times, width, label='Backward (Gradient)', color='#ff7f0e', alpha=0.8)
    
    ax_bench.set_ylabel('Time (s)')
    ax_bench.set_title('Performance Benchmark (100 steps, batch=256, dim=64)', fontweight='bold')
    ax_bench.set_xticks(x)
    ax_bench.set_xticklabels(names)
    ax_bench.legend()
    ax_bench.set_axisbelow(True)
    ax_bench.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(assets_dir, "integrators.png"), dpi=150, 
                bbox_inches='tight', facecolor='white')
    import time
    import tracemalloc

    def benchmark_integrators():
        print("\n" + "="*60)
        print(f"{'Integrator':<20} | {'Time (s)':<10} | {'Steps/sec':<10} | {'Peak Mem (MB)':<12}")
        print("-" * 60)
        
        # Benchmark setup
        dim = 100
        n_particles = 1000
        n_steps = 1000
        dt = 0.01
        device = torch.device('cpu')
        
        # Simple Harmonic force for benchmarking
        def force_fn(x):
            return -x
            
        def grad_fn(x):
            return x

        x0 = torch.randn(n_particles, dim, device=device)
        v0 = torch.randn(n_particles, dim, device=device)
        
        integrators = [
            ("OverdampedLangevin", OverdampedLangevin(gamma=1.0, kT=1.0)),
            ("BAOAB", BAOAB(gamma=1.0, kT=1.0, mass=1.0)),
            ("VelocityVerlet", VelocityVerlet(mass=1.0)),
            ("NoseHoover", NoseHoover(kT=1.0, mass=1.0, Q=1.0)),
            ("NoseHooverChain", NoseHooverChain(kT=1.0, mass=1.0, Q=1.0, n_chain=2)),
            ("ESH", ESH(eps=0.1)),
            ("GLE", GLE(kT=1.0, mass=1.0, gamma=[1.0, 2.0], c=[1.0, 2.0]))
        ]
        
        for name, integrator in integrators:
            integrator = integrator.to(device)
            
            # Helper to run integrator with correct signature
            def run_int(steps, use_warmup_slice=False):
                current_x0 = x0[:10] if use_warmup_slice else x0
                current_v0 = v0[:10] if use_warmup_slice else v0
                
                if name == "ESH":
                    integrator.run(current_x0, None, grad_fn, n_steps=steps)
                elif name == "OverdampedLangevin":
                    integrator.run(current_x0, force_fn, dt=dt, n_steps=steps)
                elif name == "VelocityVerlet":
                    # VelocityVerlet requires explicit v0
                    integrator.run(current_x0, current_v0, force_fn, dt=dt, n_steps=steps)
                else:
                    integrator.run(current_x0, None, force_fn, dt=dt, n_steps=steps)

            # Warmup
            try:
                run_int(10, use_warmup_slice=True)
            except Exception as e:
                print(f"Failed warmup for {name}: {e}")
                continue
            
            # Reset memory tracking
            tracemalloc.start()
            tracemalloc.clear_traces()
            start_mem = tracemalloc.get_traced_memory()[0]
            
            start_time = time.perf_counter()
            
            # Run benchmark
            run_int(n_steps, use_warmup_slice=False)
                
            end_time = time.perf_counter()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            duration = end_time - start_time
            # Metric: particles * steps / second
            throughput = (n_steps * n_particles) / duration
            mem_usage = (peak_mem - start_mem) / 1024 / 1024  # MB
            
            print(f"{name:<20} | {duration:<10.4f} | {throughput:<10.0f} | {mem_usage:<12.2f}")
            
        print("="*60 + "\n")

    benchmark_integrators()
