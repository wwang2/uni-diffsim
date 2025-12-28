"""Integrators for molecular dynamics.

All integrators are fully vectorized and support batch dimensions.
Convention: positions x have shape (..., dim), velocities v have shape (..., dim).
"""

import torch
from typing import Callable
import math


# Type alias for force functions: x -> force
ForceFunc = Callable[[torch.Tensor], torch.Tensor]


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
        # B: velocity half-kick
        v = v + (dt / 2) * force_fn(x) / self.mass
        # A: position half-drift
        x = x + (dt / 2) * v
        # O: Ornstein-Uhlenbeck on velocity
        alpha = math.exp(-self.gamma * dt)
        sigma = math.sqrt((self.kT / self.mass) * (1 - alpha**2))
        v = alpha * v + sigma * torch.randn_like(v)
        # A: position half-drift
        x = x + (dt / 2) * v
        # B: velocity half-kick
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
        # Velocity half-step
        v = v + (dt / 2) * force_fn(x) / self.mass
        # Position full step
        x = x + dt * v
        # Velocity half-step
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
        self.Q = Q  # thermostat mass
        self.n_chain = n_chain
    
    def step(self, x: torch.Tensor, v: torch.Tensor, xi: torch.Tensor,
             force_fn: ForceFunc, dt: float, ndof: int
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single NHC step. Returns (new_x, new_v, new_xi)."""
        # Compute kinetic energy and coupling
        KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)  # (..., 1)
        target_KE = 0.5 * ndof * self.kT
        
        # Thermostat half-step (Yoshida-Suzuki integration)
        G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
        xi_new = xi.clone()
        
        # Chain from end to start
        for j in range(self.n_chain - 1, -1, -1):
            if j == self.n_chain - 1:
                xi_new[..., j] = xi[..., j] + (dt / 4) * G
            else:
                xi_new[..., j] = xi[..., j] * torch.exp(-xi_new[..., j+1] * dt / 8)
                xi_new[..., j] = xi_new[..., j] + (dt / 4) * G
                xi_new[..., j] = xi_new[..., j] * torch.exp(-xi_new[..., j+1] * dt / 8)
            
            if j == 0:
                scale = torch.exp(-xi_new[..., 0:1] * dt / 2)
                v = v * scale
                KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
            G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q if j == 0 else \
                (self.Q * xi_new[..., j-1]**2 - self.kT) / self.Q
        
        # Velocity half-step
        v = v + (dt / 2) * force_fn(x) / self.mass
        # Position full step
        x = x + dt * v
        # Velocity half-step
        v = v + (dt / 2) * force_fn(x) / self.mass
        
        # Second thermostat half-step
        KE = 0.5 * self.mass * (v**2).sum(dim=-1, keepdim=True)
        G = (2 * KE.squeeze(-1) - ndof * self.kT) / self.Q
        
        for j in range(self.n_chain):
            if j == 0:
                scale = torch.exp(-xi_new[..., 0:1] * dt / 2)
                v = v * scale
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
        # Initialize thermostat variables
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
    
    # Apply plotting style
    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
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
    
    # Setup
    dw = DoubleWell()
    kT = 0.5
    dt = 0.01
    n_steps = 10000
    n_batch = 50
    
    def force_fn(x):
        return dw.force(x.unsqueeze(-1)).squeeze(-1)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    
    # 1. Overdamped Langevin trajectories
    ax = axes[0, 0]
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    x0 = torch.full((n_batch,), -1.0)
    traj = integrator.run(x0, force_fn, dt, n_steps, store_every=10)
    t = np.arange(traj.shape[0]) * dt * 10
    for i in range(min(5, n_batch)):
        ax.plot(t, traj[:, i].detach().numpy(), alpha=0.7, lw=0.8)
    ax.axhline(1, color='#d62728', ls='--', alpha=0.5, lw=1)
    ax.axhline(-1, color='#d62728', ls='--', alpha=0.5, lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('Overdamped Langevin', fontweight='bold')
    ax.set_axisbelow(True)
    
    # 2. BAOAB trajectories
    ax = axes[0, 1]
    integrator = BAOAB(gamma=1.0, kT=kT, mass=1.0)
    x0 = torch.full((n_batch,), -1.0)
    traj_x, traj_v = integrator.run(x0, None, force_fn, dt, n_steps, store_every=10)
    for i in range(min(5, n_batch)):
        ax.plot(t, traj_x[:, i].detach().numpy(), alpha=0.7, lw=0.8)
    ax.axhline(1, color='#d62728', ls='--', alpha=0.5, lw=1)
    ax.axhline(-1, color='#d62728', ls='--', alpha=0.5, lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('BAOAB (Underdamped)', fontweight='bold')
    ax.set_axisbelow(True)
    
    # 3. Distribution comparison
    ax = axes[1, 0]
    samples = {}
    
    # Collect samples from burn-in onwards
    burn_in = 2000
    for name, traj_data in [('Overdamped', traj), ('BAOAB', traj_x)]:
        samples[name] = traj_data[burn_in//10:].flatten().detach().numpy()
    
    # Plot histograms
    x_range = (-2.5, 2.5)
    bins = 60
    colors = {'Overdamped': '#1f77b4', 'BAOAB': '#ff7f0e'}
    for name, s in samples.items():
        ax.hist(s, bins=bins, range=x_range, density=True, alpha=0.5, 
                label=name, color=colors[name])
    
    # Theoretical Boltzmann
    x_th = torch.linspace(-2.5, 2.5, 200)
    u_th = dw.energy(x_th).detach().numpy()
    p_th = np.exp(-u_th / kT)
    p_th = p_th / (p_th.sum() * (x_th[1] - x_th[0]).item())
    ax.plot(x_th.numpy(), p_th, 'k-', lw=2, label='Boltzmann')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('Sampling Distribution', fontweight='bold')
    ax.legend()
    ax.set_axisbelow(True)
    
    # 4. Energy conservation (Verlet vs BAOAB)
    ax = axes[1, 1]
    harm = Harmonic(k=1.0)
    def harm_force(x):
        return harm.force(x)
    
    x0_2d = torch.tensor([[1.0, 0.0]])
    v0_2d = torch.tensor([[0.0, 1.0]])
    dt_fine = 0.05
    n_steps_energy = 2000
    
    # Velocity Verlet (should conserve energy)
    verlet = VelocityVerlet(mass=1.0)
    traj_x_v, traj_v_v = verlet.run(x0_2d, v0_2d, harm_force, dt_fine, n_steps_energy)
    E_verlet = harm.energy(traj_x_v.squeeze(1)) + kinetic_energy(traj_v_v.squeeze(1))
    
    # BAOAB (dissipates to thermal equilibrium)
    baoab = BAOAB(gamma=0.1, kT=0.5, mass=1.0)
    traj_x_b, traj_v_b = baoab.run(x0_2d, v0_2d, harm_force, dt_fine, n_steps_energy)
    E_baoab = harm.energy(traj_x_b.squeeze(1)) + kinetic_energy(traj_v_b.squeeze(1))
    
    t_energy = np.arange(n_steps_energy + 1) * dt_fine
    ax.plot(t_energy, E_verlet.detach().numpy(), label='Verlet (NVE)', lw=1.5)
    ax.plot(t_energy, E_baoab.detach().numpy(), label='BAOAB (NVT)', lw=1.5, alpha=0.8)
    ax.axhline(0.5, color='#2ca02c', ls='--', alpha=0.7, lw=1, label=f'kT={baoab.kT}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy: Verlet vs BAOAB', fontweight='bold')
    ax.legend()
    ax.set_axisbelow(True)
    
    plt.savefig(os.path.join(assets_dir, "integrators.png"), dpi=150, 
                bbox_inches='tight', facecolor='white')
    print(f"Saved integrators plot to assets/integrators.png")

