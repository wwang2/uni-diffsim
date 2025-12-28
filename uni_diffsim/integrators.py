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
"""

import torch
from typing import Callable
import math


ForceFunc = Callable[[torch.Tensor], torch.Tensor]
GradFunc = Callable[[torch.Tensor], torch.Tensor]  # For ESH: gradient of energy


class OverdampedLangevin:
    """Overdamped Langevin dynamics: dx = F/γ dt + √(2kT/γ) dW.
    
    High-friction limit where inertia is negligible.
    Samples from Boltzmann distribution p(x) ∝ exp(-U(x)/kT).
    """
    
    def __init__(self, gamma: float = 1.0, kT: float = 1.0):
        self.gamma = gamma
        self.kT = kT
    
    def step(self, x: torch.Tensor, force_fn: ForceFunc, dt: float) -> torch.Tensor:
        """Single integration step. Returns new positions."""
        force = force_fn(x)
        noise_scale = math.sqrt(2 * self.kT * dt / self.gamma)
        return x + (force / self.gamma) * dt + noise_scale * torch.randn_like(x)
    
    def run(self, x0: torch.Tensor, force_fn: ForceFunc, dt: float, 
            n_steps: int, store_every: int = 1) -> torch.Tensor:
        """Run trajectory. Returns (n_stored, ..., dim) positions."""
        x = x0
        n_stored = n_steps // store_every + 1
        traj = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj[0] = x0
        idx = 1
        for i in range(1, n_steps + 1):
            x = self.step(x, force_fn, dt)
            if i % store_every == 0:
                traj[idx] = x
                idx += 1
        return traj


class BAOAB:
    """BAOAB splitting for underdamped Langevin dynamics.
    
    B: velocity kick (half), A: position drift (half), O: Ornstein-Uhlenbeck noise,
    A: position drift (half), B: velocity kick (half).
    
    Excellent sampling properties with low discretization error.
    """
    
    def __init__(self, gamma: float = 1.0, kT: float = 1.0, mass: float = 1.0):
        self.gamma = gamma
        self.kT = kT
        self.mass = mass
    
    def step(self, x: torch.Tensor, v: torch.Tensor, force_fn: ForceFunc, 
             dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Single BAOAB step. Returns (new_x, new_v)."""
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + (dt / 2) * v
        alpha = math.exp(-self.gamma * dt)
        sigma = math.sqrt((self.kT / self.mass) * (1 - alpha**2))
        v = alpha * v + sigma * torch.randn_like(v)
        x = x + (dt / 2) * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        return x, v
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities) each (n_stored, ...)."""
        x = x0
        v = v0 if v0 is not None else torch.randn_like(x0) * math.sqrt(self.kT / self.mass)
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_v = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_x[0], traj_v[0] = x0, v
        idx = 1
        for i in range(1, n_steps + 1):
            x, v = self.step(x, v, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx], traj_v[idx] = x, v
                idx += 1
        return traj_x, traj_v


class VelocityVerlet:
    """Symplectic velocity Verlet integrator (NVE ensemble).
    
    Preserves phase-space volume and has excellent energy conservation.
    No thermostat - samples microcanonical ensemble.
    """
    
    def __init__(self, mass: float = 1.0):
        self.mass = mass
    
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
        traj_x = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_v = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_x[0], traj_v[0] = x0, v0
        idx = 1
        for i in range(1, n_steps + 1):
            x, v = self.step(x, v, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx], traj_v[idx] = x, v
                idx += 1
        return traj_x, traj_v


class NoseHooverChain:
    """Nosé-Hoover chain thermostat for deterministic canonical sampling.
    
    Extends phase space with thermostat variables to sample NVT ensemble
    without stochastic noise. Chain length > 1 improves ergodicity.
    """
    
    def __init__(self, kT: float = 1.0, mass: float = 1.0, 
                 Q: float = 1.0, n_chain: int = 2):
        self.kT = kT
        self.mass = mass
        self.Q = Q
        self.n_chain = n_chain
    
    def step(self, x: torch.Tensor, v: torch.Tensor, xi: torch.Tensor,
             force_fn: ForceFunc, dt: float, ndof: int
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single NHC step. Returns (new_x, new_v, new_xi)."""
        KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
        G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
        xi_new = xi.clone()
        
        for j in range(self.n_chain - 1, -1, -1):
            if j == self.n_chain - 1:
                xi_new[..., j] = xi[..., j] + (dt / 4) * G
            else:
                xi_new[..., j] = xi[..., j] * torch.exp(-xi_new[..., j+1] * dt / 8)
                xi_new[..., j] = xi_new[..., j] + (dt / 4) * G
                xi_new[..., j] = xi_new[..., j] * torch.exp(-xi_new[..., j+1] * dt / 8)
            if j == 0:
                v = v * torch.exp(-xi_new[..., 0:1] * dt / 2)
                KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
            G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q if j == 0 else \
                (self.Q * xi_new[..., j-1]**2 - self.kT) / self.Q
        
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass
        
        KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
        G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
        
        for j in range(self.n_chain):
            if j == 0:
                v = v * torch.exp(-xi_new[..., 0:1] * dt / 2)
                KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
                G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
            if j == self.n_chain - 1:
                xi_new[..., j] = xi_new[..., j] + (dt / 4) * G
            else:
                xi_new[..., j] = xi_new[..., j] * torch.exp(-xi_new[..., j+1] * dt / 8)
                xi_new[..., j] = xi_new[..., j] + (dt / 4) * G
                xi_new[..., j] = xi_new[..., j] * torch.exp(-xi_new[..., j+1] * dt / 8)
            G = (self.Q * xi_new[..., j]**2 - self.kT) / self.Q
        
        return x, v, xi_new
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trajectory. Returns (positions, velocities)."""
        x = x0
        v = v0 if v0 is not None else torch.randn_like(x0) * math.sqrt(self.kT / self.mass)
        ndof = x0.shape[-1]
        xi_shape = x0.shape[:-1] + (self.n_chain,)
        xi = torch.zeros(xi_shape, device=x0.device, dtype=x0.dtype)
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_v = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_x[0], traj_v[0] = x0, v
        idx = 1
        for i in range(1, n_steps + 1):
            x, v, xi = self.step(x, v, xi, force_fn, dt, ndof)
            if i % store_every == 0:
                traj_x[idx], traj_v[idx] = x, v
                idx += 1
        return traj_x, traj_v


class ESH:
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
    """
    
    def __init__(self, eps: float = 0.1):
        self.eps = eps
    
    def _u_r_step(self, u: torch.Tensor, r: torch.Tensor, 
                  grad: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Update unit velocity u and log-magnitude r. Vectorized over batch."""
        # Dimension d from last axis
        d = u.shape[-1]
        
        # Gradient norm and unit vector
        g_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-10)
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
        traj_x = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_u = torch.empty((n_stored,) + x0.shape, device=x0.device, dtype=x0.dtype)
        traj_r = torch.empty((n_stored,) + r.shape, device=x0.device, dtype=x0.dtype)
        traj_x[0], traj_u[0], traj_r[0] = x0, u, r
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, u, r = self.step(x, u, r, grad_fn, eps)
            if i % store_every == 0:
                traj_x[idx], traj_u[idx], traj_r[idx] = x, u, r
                idx += 1
        
        return traj_x, traj_u, traj_r


class GLE:
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
    """
    
    def __init__(self, kT: float = 1.0, mass: float = 1.0,
                 gamma: list[float] | None = None,
                 c: list[float] | None = None):
        """
        Args:
            kT: Temperature
            mass: Particle mass  
            gamma: Decay rates for memory kernel modes
            c: Coupling strengths for each mode
        """
        self.kT = kT
        self.mass = mass
        
        # Default: single mode (standard Langevin-like)
        if gamma is None:
            gamma = [1.0]
        if c is None:
            c = list(gamma)
            
        self.gamma = gamma
        self.c = c
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
        device, dtype = x.device, x.dtype
        gamma = torch.tensor(self.gamma, device=device, dtype=dtype)
        c = torch.tensor(self.c, device=device, dtype=dtype)
        
        # Friction from auxiliary variables
        friction = s.sum(dim=-1)
        
        # B: Half-step velocity
        v = v + (dt / 2) * (force_fn(x) - friction) / self.mass
        
        # A: Half-step position
        x = x + (dt / 2) * v
        
        # O: Update auxiliary variables (vectorized)
        alpha = torch.exp(-gamma * dt)
        # Noise variance from fluctuation-dissipation
        sigma = torch.sqrt(c * self.kT * (1 - alpha**2))
        noise = torch.randn_like(s) * sigma
        
        # Drift coefficient
        drift_coef = (c / gamma) * (1 - alpha)
        s = alpha * s + drift_coef * v.unsqueeze(-1) + noise
        
        # A: Half-step position
        x = x + (dt / 2) * v
        
        # B: Half-step velocity with updated friction
        friction = s.sum(dim=-1)
        v = v + (dt / 2) * (force_fn(x) - friction) / self.mass
        
        return x, v, s
    
    def run(self, x0: torch.Tensor, v0: torch.Tensor | None, force_fn: ForceFunc,
            dt: float, n_steps: int, store_every: int = 1
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GLE trajectory. Returns (positions, velocities)."""
        device, dtype = x0.device, x0.dtype
        x = x0
        v = v0 if v0 is not None else torch.randn_like(x0) * math.sqrt(self.kT / self.mass)
        
        # Initialize auxiliary variables at thermal equilibrium
        c = torch.tensor(self.c, device=device, dtype=dtype)
        s_shape = x0.shape + (self.n_modes,)
        s = torch.randn(s_shape, device=device, dtype=dtype) * torch.sqrt(c * self.kT)
        
        n_stored = n_steps // store_every + 1
        traj_x = torch.empty((n_stored,) + x0.shape, device=device, dtype=dtype)
        traj_v = torch.empty((n_stored,) + x0.shape, device=device, dtype=dtype)
        traj_x[0], traj_v[0] = x0, v
        
        idx = 1
        for i in range(1, n_steps + 1):
            x, v, s = self.step(x, v, s, force_fn, dt)
            if i % store_every == 0:
                traj_x[idx], traj_v[idx] = x, v
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
    from .potentials import DoubleWell, Harmonic
    
    # Plotting style
    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 8.0,
        "axes.labelpad": 4.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    
    # Setup for 1D double well
    dw = DoubleWell()
    kT = 0.5
    dt = 0.01
    n_steps = 10000
    n_batch = 50
    
    def force_fn_1d(x):
        return dw.force(x.unsqueeze(-1)).squeeze(-1)
    
    # 1. Overdamped Langevin
    ax = axes[0, 0]
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    x0 = torch.full((n_batch,), -1.0)
    traj_od = integrator.run(x0, force_fn_1d, dt, n_steps, store_every=10)
    t = np.arange(traj_od.shape[0]) * dt * 10
    for i in range(min(3, n_batch)):
        ax.plot(t, traj_od[:, i].detach().numpy(), alpha=0.7, lw=0.7)
    ax.axhline(1, color='#d62728', ls='--', alpha=0.4, lw=1)
    ax.axhline(-1, color='#d62728', ls='--', alpha=0.4, lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('Overdamped Langevin', fontweight='bold')
    ax.set_axisbelow(True)
    
    # 2. BAOAB
    ax = axes[0, 1]
    integrator = BAOAB(gamma=1.0, kT=kT, mass=1.0)
    traj_baoab, _ = integrator.run(x0, None, force_fn_1d, dt, n_steps, store_every=10)
    for i in range(min(3, n_batch)):
        ax.plot(t, traj_baoab[:, i].detach().numpy(), alpha=0.7, lw=0.7)
    ax.axhline(1, color='#d62728', ls='--', alpha=0.4, lw=1)
    ax.axhline(-1, color='#d62728', ls='--', alpha=0.4, lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('BAOAB', fontweight='bold')
    ax.set_axisbelow(True)
    
    # 3. GLE (colored noise)
    ax = axes[0, 2]
    gle = GLE(kT=kT, mass=1.0, gamma=[0.5, 2.0], c=[0.3, 1.0])
    traj_gle, _ = gle.run(x0, None, force_fn_1d, dt, n_steps, store_every=10)
    for i in range(min(3, n_batch)):
        ax.plot(t, traj_gle[:, i].detach().numpy(), alpha=0.7, lw=0.7)
    ax.axhline(1, color='#d62728', ls='--', alpha=0.4, lw=1)
    ax.axhline(-1, color='#d62728', ls='--', alpha=0.4, lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('GLE (colored noise)', fontweight='bold')
    ax.set_axisbelow(True)
    
    # 4. Distribution comparison (1D stochastic methods)
    ax = axes[1, 0]
    burn_in = 2000
    samples = {
        'Overdamped': traj_od[burn_in//10:].flatten().detach().numpy(),
        'BAOAB': traj_baoab[burn_in//10:].flatten().detach().numpy(),
        'GLE': traj_gle[burn_in//10:].flatten().detach().numpy(),
    }
    colors = {'Overdamped': '#1f77b4', 'BAOAB': '#ff7f0e', 'GLE': '#2ca02c'}
    for name, s in samples.items():
        ax.hist(s, bins=50, range=(-2.5, 2.5), density=True, alpha=0.4, 
                label=name, color=colors[name])
    
    x_th = torch.linspace(-2.5, 2.5, 200)
    u_th = dw.energy(x_th).detach().numpy()
    p_th = np.exp(-u_th / kT)
    p_th = p_th / (p_th.sum() * (x_th[1] - x_th[0]).item())
    ax.plot(x_th.numpy(), p_th, 'k-', lw=2, label='Boltzmann')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('Sampling Distribution', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_axisbelow(True)
    
    # 5. ESH ergodic sampling on 2D Harmonic
    ax = axes[1, 1]
    harm_2d = Harmonic(k=1.0)
    
    def grad_harm_2d(x):
        return -harm_2d.force(x)
    
    # ESH samples ergodically along a single trajectory
    esh = ESH(eps=0.2)
    x0_2d = torch.tensor([[1.0, 0.0]])
    u0_2d = torch.tensor([[0.0, 1.0]])
    
    traj_esh_2d, _, _ = esh.run(x0_2d, u0_2d, grad_harm_2d, n_steps=10000, 
                                 dt=0.2, store_every=1)
    traj_np = traj_esh_2d.squeeze(1).detach().numpy()
    
    # Compare with BAOAB
    baoab_2d = BAOAB(gamma=1.0, kT=1.0, mass=1.0)
    traj_baoab_2d, _ = baoab_2d.run(
        torch.zeros(500, 2), None, harm_2d.force, dt=0.05, n_steps=2000, store_every=20
    )
    baoab_samples = traj_baoab_2d[-1].detach().numpy()
    
    ax.scatter(traj_np[1000:, 0], traj_np[1000:, 1], s=1, alpha=0.3, 
               c='#9467bd', label='ESH (ergodic)')
    ax.scatter(baoab_samples[:, 0], baoab_samples[:, 1], s=15, alpha=0.5,
               c='#ff7f0e', edgecolor='none', label='BAOAB')
    
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [1.0, 1.5, 2.0]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.3, lw=0.8)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('ESH vs BAOAB (2D Harmonic)', fontweight='bold')
    ax.legend(loc='upper right', markerscale=3)
    ax.set_axisbelow(True)
    
    # 6. Energy conservation comparison
    ax = axes[1, 2]
    harm = Harmonic(k=1.0)
    def harm_force(x):
        return harm.force(x)
    def harm_grad(x):
        return -harm_force(x)
    
    x0_h = torch.tensor([[1.0, 0.0]])
    v0_h = torch.tensor([[0.0, 1.0]])
    dt_fine = 0.05
    n_energy = 1000
    
    # Verlet
    verlet = VelocityVerlet(mass=1.0)
    tx_v, tv_v = verlet.run(x0_h, v0_h, harm_force, dt_fine, n_energy)
    E_verlet = harm.energy(tx_v.squeeze(1)) + kinetic_energy(tv_v.squeeze(1))
    
    # ESH (conserves modified Hamiltonian)
    esh_h = ESH(eps=dt_fine)
    tx_e, tu_e, tr_e = esh_h.run(x0_h, None, harm_grad, n_energy, dt=dt_fine)
    # ESH Hamiltonian: H = E(x) + d/2 * log(|v|²/d) = E(x) + r + const
    E_esh = harm.energy(tx_e.squeeze(1)) + tr_e.squeeze(1)
    
    # NHC
    nhc = NoseHooverChain(kT=0.5, mass=1.0, Q=1.0)
    tx_n, tv_n = nhc.run(x0_h, v0_h, harm_force, dt_fine, n_energy)
    E_nhc = harm.energy(tx_n.squeeze(1)) + kinetic_energy(tv_n.squeeze(1))
    
    t_e = np.arange(n_energy + 1) * dt_fine
    ax.plot(t_e, E_verlet.detach().numpy(), label='Verlet', lw=1.2)
    ax.plot(t_e, E_esh.detach().numpy(), label='ESH (H_ESH)', lw=1.2, alpha=0.8)
    ax.plot(t_e, E_nhc.detach().numpy(), label='NHC', lw=1.2, alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Hamiltonian')
    ax.set_title('Energy Conservation', fontweight='bold')
    ax.legend()
    ax.set_axisbelow(True)
    
    plt.savefig(os.path.join(assets_dir, "integrators.png"), dpi=150, 
                bbox_inches='tight', facecolor='white')
    print(f"Saved integrators plot to assets/integrators.png")
