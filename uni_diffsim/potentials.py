"""Potential energy surfaces for benchmark systems.

All potentials are fully vectorized and support arbitrary batch dimensions.
"""

import torch
import torch.nn as nn


class Potential(nn.Module):
    """Base class for potentials. Subclasses must implement energy()."""
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy. Override in subclass."""
        raise NotImplementedError
    
    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Compute force = -grad(U). Works for any batch shape."""
        x = x.detach().requires_grad_(True)
        u = self.energy(x)
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        return -grad
    
    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hessian d²U/dx². Returns (..., d, d) for input (..., d)."""
        x = x.detach().requires_grad_(True)
        u = self.energy(x)
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d = x.shape[-1]
        hess_rows = []
        for i in range(d):
            h_row = torch.autograd.grad(grad[..., i].sum(), x, retain_graph=True)[0]
            hess_rows.append(h_row)
        return torch.stack(hess_rows, dim=-2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy(x)


class DoubleWell(Potential):
    """1D Double well: U(x) = a*(x² - 1)².
    
    Minima at x = ±1, barrier at x = 0 with height a.
    Input shape: (...,) or (..., 1). Output shape: (...,).
    """
    
    def __init__(self, barrier_height: float = 1.0):
        super().__init__()
        self.barrier_height = barrier_height
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1:] == (1,):
            x = x.squeeze(-1)
        return self.barrier_height * (x**2 - 1)**2


class MullerBrown(Potential):
    """2D Müller-Brown potential - classic reaction path benchmark.
    
    Has 3 minima and 2 saddle points.
    Input shape: (..., 2). Output shape: (...,).
    """
    
    # Standard parameters (registered as buffers for device handling)
    _A = torch.tensor([-200., -100., -170., 15.])
    _a = torch.tensor([-1., -1., -6.5, 0.7])
    _b = torch.tensor([0., 0., 11., 0.6])
    _c = torch.tensor([-10., -10., -6.5, 0.7])
    _x0 = torch.tensor([1., 0., -0.5, -1.])
    _y0 = torch.tensor([0., 0.5, 1.5, 1.])
    
    def __init__(self):
        super().__init__()
        self.register_buffer("A", self._A.clone())
        self.register_buffer("a", self._a.clone())
        self.register_buffer("b", self._b.clone())
        self.register_buffer("c", self._c.clone())
        self.register_buffer("x0", self._x0.clone())
        self.register_buffer("y0", self._y0.clone())
    
    def energy(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (..., 2) -> (...)"""
        x, y = xy[..., 0], xy[..., 1]
        # Vectorized over all 4 terms: expand for broadcasting
        # x: (...), x0: (4,) -> dx: (..., 4)
        dx = x.unsqueeze(-1) - self.x0
        dy = y.unsqueeze(-1) - self.y0
        exponent = self.a * dx**2 + self.b * dx * dy + self.c * dy**2
        return (self.A * torch.exp(exponent)).sum(-1)


class LennardJones(Potential):
    """N-particle Lennard-Jones potential.
    
    U = 4ε Σ_{i<j} [(σ/r_ij)¹² - (σ/r_ij)⁶]
    
    Input shape: (..., n_particles, dim). Output shape: (...,).
    """
    
    def __init__(self, eps: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.eps = eps
        self.sigma = sigma
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_particles, dim) -> (...)"""
        # Pairwise distances: (..., n, n, dim)
        diff = x.unsqueeze(-2) - x.unsqueeze(-3)
        r2 = (diff**2).sum(-1)  # (..., n, n)
        
        # Upper triangular mask (i < j pairs only)
        n = x.shape[-2]
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=x.device)
        r2_pairs = r2[..., idx_i, idx_j]  # (..., n_pairs)
        
        # LJ: 4ε[(σ/r)¹² - (σ/r)⁶]
        s2 = self.sigma**2 / r2_pairs
        s6 = s2**3
        u_pairs = 4 * self.eps * (s6**2 - s6)
        return u_pairs.sum(-1)


class Harmonic(Potential):
    """Simple harmonic oscillator: U(x) = 0.5 * k * ||x - x0||².
    
    Input shape: (..., d). Output shape: (...,).
    """
    
    def __init__(self, k: float = 1.0, center: torch.Tensor | None = None):
        super().__init__()
        self.k = k
        if center is not None:
            self.register_buffer("center", center)
        else:
            self.center = None
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if self.center is not None:
            x = x - self.center
        return 0.5 * self.k * (x**2).sum(-1)


if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    
    # 1. Double well energy and force
    ax = axes[0, 0]
    dw = DoubleWell()
    x = torch.linspace(-2, 2, 200)
    u = dw.energy(x)
    f = dw.force(x.unsqueeze(-1)).squeeze()
    ax.plot(x.numpy(), u.detach().numpy(), 'k-', lw=2, label='U(x)')
    ax.plot(x.numpy(), f.detach().numpy(), '#d62728', ls='--', lw=1.5, label='F(x)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('Energy / Force')
    ax.set_title('Double Well', fontweight='bold')
    ax.legend()
    ax.set_axisbelow(True)
    
    # 2. Müller-Brown contour
    ax = axes[0, 1]
    mb = MullerBrown()
    x = torch.linspace(-1.5, 1.2, 100)
    y = torch.linspace(-0.5, 2.0, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X, Y], dim=-1)
    U = mb.energy(xy)
    levels = np.linspace(-150, 100, 30)
    cs = ax.contourf(X.numpy(), Y.numpy(), U.detach().numpy(), levels=levels, cmap='viridis')
    ax.contour(X.numpy(), Y.numpy(), U.detach().numpy(), levels=levels, colors='k', linewidths=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Müller-Brown', fontweight='bold')
    plt.colorbar(cs, ax=ax, label='U')
    ax.set_axisbelow(True)
    
    # 3. LJ-7 cluster
    ax = axes[1, 0]
    lj = LennardJones()
    angles = torch.linspace(0, 2*np.pi, 7)[:-1]
    r = 1.12
    positions = torch.zeros(7, 2)
    positions[1:, 0] = r * torch.cos(angles)
    positions[1:, 1] = r * torch.sin(angles)
    ax.scatter(positions[:, 0].numpy(), positions[:, 1].numpy(), 
               s=500, c='#1f77b4', edgecolor='k', lw=1.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(f'LJ-7 cluster (U={lj.energy(positions).item():.2f})', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_axisbelow(True)
    
    # 4. Batch computation demo
    ax = axes[1, 1]
    batch_sizes = [1, 10, 100, 1000, 10000]
    times = []
    import time
    for bs in batch_sizes:
        xy_batch = torch.randn(bs, 2)
        _ = mb.energy(xy_batch)
        start = time.perf_counter()
        for _ in range(100):
            _ = mb.energy(xy_batch)
        times.append((time.perf_counter() - start) / 100 * 1000)
    
    ax.loglog(batch_sizes, times, 'o-', lw=2, markersize=8, color='#2ca02c')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Vectorization Scaling', fontweight='bold')
    ax.set_axisbelow(True)
    
    plt.savefig(os.path.join(assets_dir, "potentials.png"), dpi=150, 
                bbox_inches='tight', facecolor='white')
    print(f"Saved potentials plot to assets/potentials.png")

