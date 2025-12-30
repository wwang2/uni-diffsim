"""Potential energy surfaces for benchmark systems.

All potentials are fully vectorized and support arbitrary batch dimensions.
"""

import torch
import torch.nn as nn
from torch.func import hessian, vmap


class Potential(nn.Module):
    """Base class for potentials. Subclasses must implement energy()."""
    
    @property
    def event_dim(self) -> int:
        """Number of dimensions per event (system state).
        Default is 1 (vector state). Subclasses like LennardJones (particles) should override.
        """
        return 1
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy. Override in subclass."""
        raise NotImplementedError
    
    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Compute force = -grad(U). Works for any batch shape."""
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            u = self.energy(x)
            grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        return -grad
    
    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hessian d²U/dx². Returns (..., d, d) for input (..., d).
        
        Uses torch.func for efficient, fully vectorized computation without loops.
        """
        ndim = x.ndim
        event_dim = self.event_dim
        
        # Check if we have batch dimensions
        if ndim < event_dim:
            # Should not happen for valid inputs given event_dim
             raise ValueError(f"Input dimension {ndim} smaller than event dimension {event_dim}")

        is_batched = ndim > event_dim
        
        if not is_batched:
            # Single sample: (d1, ..., dk)
            return hessian(self.energy)(x)
            
        # Batched case: flatten batch dims -> (N, d1, ..., dk)
        batch_shape = x.shape[:-event_dim]
        event_shape = x.shape[-event_dim:]
        x_flat = x.reshape(-1, *event_shape)
        
        def energy_wrapper(x_in):
            # x_in has shape event_shape
            # Ensure scalar output for hessian
            return self.energy(x_in).squeeze()
            
        # vmap over batch dimension (0)
        h = vmap(hessian(energy_wrapper))(x_flat)
        
        # Reshape back: (*batch_shape, *event_shape, *event_shape)
        return h.view(*batch_shape, *event_shape, *event_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy(x)


class DoubleWell(Potential):
    """1D Double well: U(x) = a*(x² - 1)².
    
    Minima at x = ±1, barrier at x = 0 with height a.
    Input shape: (...,) or (..., 1). Output shape: (...,).
    
    Args:
        barrier_height: Height of the barrier at x=0. Differentiable parameter.
    """
    
    def __init__(self, barrier_height: float = 1.0):
        super().__init__()
        self.barrier_height = nn.Parameter(torch.tensor(barrier_height))
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1:] == (1,):
            x = x.squeeze(-1)
        return self.barrier_height * (x**2 - 1)**2


class AsymmetricDoubleWell(Potential):
    """1D Asymmetric double well: U(x) = a*(x² - 1)² + b*x.

    A tilted double-well potential where:
    - a controls the barrier height
    - b controls the asymmetry (tilt)

    When b > 0, the left well (x ≈ -1) is lower.
    When b < 0, the right well (x ≈ +1) is lower.

    This potential is interesting for studying:
    - Population ratios between metastable states
    - Transition rates and their parameter dependence
    - Gradient of occupation probabilities w.r.t. asymmetry

    At equilibrium, the ratio of populations follows:
        P_right / P_left ≈ exp(-β * 2b)  (for small b, high barrier)

    Input shape: (...,) or (..., 1). Output shape: (...,).

    Args:
        barrier_height: Height of the barrier (parameter a). Differentiable.
        asymmetry: Tilt of the potential (parameter b). Differentiable.
    """

    def __init__(self, barrier_height: float = 1.0, asymmetry: float = 0.0):
        super().__init__()
        self.barrier_height = nn.Parameter(torch.tensor(barrier_height))
        self.asymmetry = nn.Parameter(torch.tensor(asymmetry))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1:] == (1,):
            x = x.squeeze(-1)
        return self.barrier_height * (x**2 - 1)**2 + self.asymmetry * x

    def well_depths(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return approximate energy of left and right wells.

        For small asymmetry, wells are near x = ±1.
        Returns (U_left, U_right).
        """
        # Approximate well positions (exact for b=0)
        x_left = torch.tensor(-1.0)
        x_right = torch.tensor(1.0)
        return self.energy(x_left), self.energy(x_right)


class DoubleWell2D(Potential):
    """2D Double well: U(x,y) = a*(x² - 1)² + b*y².
    
    Two minima at (±1, 0), connected by a saddle at (0, 0).
    The y-direction is a simple harmonic potential.
    
    Input shape: (..., 2). Output shape: (...,).
    
    Args:
        barrier_height: Height of the barrier in x-direction. Differentiable parameter.
        k_y: Spring constant in y-direction. Differentiable parameter.
    """
    
    def __init__(self, barrier_height: float = 1.0, k_y: float = 1.0):
        super().__init__()
        self.barrier_height = nn.Parameter(torch.tensor(barrier_height))
        self.k_y = nn.Parameter(torch.tensor(k_y))
    
    def energy(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[..., 0], xy[..., 1]
        return self.barrier_height * (x**2 - 1)**2 + 0.5 * self.k_y * y**2


class MullerBrown(Potential):
    """2D Müller-Brown potential - classic reaction path benchmark.
    
    Has 3 minima and 2 saddle points.
    Input shape: (..., 2). Output shape: (...,).
    
    All parameters are differentiable. The default values correspond to the
    standard Müller-Brown potential.
    """
    
    # Standard parameter values
    _A_default = torch.tensor([-200., -100., -170., 15.])
    _a_default = torch.tensor([-1., -1., -6.5, 0.7])
    _b_default = torch.tensor([0., 0., 11., 0.6])
    _c_default = torch.tensor([-10., -10., -6.5, 0.7])
    _x0_default = torch.tensor([1., 0., -0.5, -1.])
    _y0_default = torch.tensor([0., 0.5, 1.5, 1.])
    
    def __init__(self, 
                 A: torch.Tensor | None = None,
                 a: torch.Tensor | None = None,
                 b: torch.Tensor | None = None,
                 c: torch.Tensor | None = None,
                 x0: torch.Tensor | None = None,
                 y0: torch.Tensor | None = None):
        super().__init__()
        self.A = nn.Parameter(A.clone() if A is not None else self._A_default.clone())
        self.a = nn.Parameter(a.clone() if a is not None else self._a_default.clone())
        self.b = nn.Parameter(b.clone() if b is not None else self._b_default.clone())
        self.c = nn.Parameter(c.clone() if c is not None else self._c_default.clone())
        self.x0 = nn.Parameter(x0.clone() if x0 is not None else self._x0_default.clone())
        self.y0 = nn.Parameter(y0.clone() if y0 is not None else self._y0_default.clone())
    
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
    """N-particle Lennard-Jones potential with optional periodic boundary conditions.
    
    U = 4ε Σ_{i<j} [(σ/r_ij)¹² - (σ/r_ij)⁶]
    
    For PBC, uses minimum image convention: distances are wrapped to [-L/2, L/2].
    
    Input shape: (..., n_particles, dim). Output shape: (...,).
    
    Args:
        eps: LJ well depth (default 1.0). Differentiable parameter.
        sigma: LJ length scale (default 1.0). Differentiable parameter.
        box_size: Simulation box size for PBC. Can be:
            - None: no periodic boundaries (default)
            - float: cubic/square box with side length L
            - Tensor of shape (dim,): rectangular box with different lengths per dimension
            Note: box_size is a buffer (not differentiable) since it defines geometry.
    """
    
    def __init__(self, eps: float = 1.0, sigma: float = 1.0, 
                 box_size: float | torch.Tensor | None = None):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        
        if box_size is not None:
            if isinstance(box_size, (int, float)):
                box_size = torch.tensor([box_size])
            elif not isinstance(box_size, torch.Tensor):
                box_size = torch.tensor(box_size)
            self.register_buffer("box_size", box_size.float())
        else:
            self.box_size = None
            
    @property
    def event_dim(self) -> int:
        return 2
    
    def _minimum_image(self, diff: torch.Tensor) -> torch.Tensor:
        """Apply minimum image convention for periodic boundaries.
        
        diff: (..., dim) displacement vectors
        Returns: wrapped displacements in [-L/2, L/2]
        """
        if self.box_size is None:
            return diff
        # Wrap to [-L/2, L/2] using remainder
        # box_size broadcasts: (dim,) with diff (..., dim)
        return diff - self.box_size * torch.round(diff / self.box_size)
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_particles, dim) -> (...)"""
        n = x.shape[-2]
        
        # Get indices for upper triangular part (i < j)
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=x.device)
        
        # Advanced indexing to gather particle positions for pairs
        # x: (..., n, dim) -> xi, xj: (..., n_pairs, dim)
        xi = x[..., idx_i, :]
        xj = x[..., idx_j, :]
        
        # Compute pair difference vectors directly
        diff = xi - xj
        
        # Apply minimum image convention for PBC
        diff = self._minimum_image(diff)
        
        # Squared distances
        r2_pairs = (diff**2).sum(-1)  # (..., n_pairs)
        
        # LJ: 4ε[(σ/r)¹² - (σ/r)⁶]
        s2 = self.sigma**2 / r2_pairs
        s6 = s2**3
        u_pairs = 4 * self.eps * (s6**2 - s6)
        return u_pairs.sum(-1)


class LennardJonesVerlet(LennardJones):
    """Lennard-Jones with Verlet Neighbor Lists to avoid O(N²) scaling.
    
    Uses a sparse neighbor list that is updated only when necessary (not handled automatically here).
    The user must call `update_neighbor_list(x)` periodically.
    
    Args:
        eps: LJ well depth.
        sigma: LJ length scale.
        box_size: PBC box size.
        cutoff: Cutoff distance for interaction (default 2.5*sigma).
        skin: Skin distance for neighbor list buffer (default 0.5*sigma).
    """
    
    def __init__(self, eps: float = 1.0, sigma: float = 1.0, 
                 box_size: float | torch.Tensor | None = None,
                 cutoff: float = 2.5, skin: float = 0.5):
        super().__init__(eps, sigma, box_size)
        self.cutoff = cutoff
        self.skin = skin
        self.r_cut_skin = cutoff + skin
        
        # Buffers for neighbor list
        self.register_buffer("neighbor_list_i", torch.empty(0, dtype=torch.long))
        self.register_buffer("neighbor_list_j", torch.empty(0, dtype=torch.long))
        
        # Store last update position for displacement check (optional usage)
        self.register_buffer("last_update_x", torch.zeros(1))
        
    def update_neighbor_list(self, x: torch.Tensor):
        """Rebuild the neighbor list. O(N²) operation.
        
        Finds all pairs with distance < cutoff + skin.
        """
        with torch.no_grad():
            # Ensure we work with flattened batch or just the last 2 dims if possible.
            # Neighbor lists for batched inputs are tricky (indices differ per batch).
            # This implementation assumes single configuration or shared neighbors (unlikely).
            # STRICT LIMITATION: Currently supports only single system (batch_dim=0).
            if x.ndim > 2:
                raise NotImplementedError("Batched Verlet lists not yet supported. Use scalar batch.")
                
            n = x.shape[0]
            device = x.device
            
            # Compute all pairwise distances (amortized cost)
            if self.box_size is None:
                # Efficiently use cdist for non-PBC
                dists = torch.cdist(x, x)
            else:
                # Manual PBC distance (O(N^2) memory!)
                # Note: For very large N, this O(N^2) memory might OOM.
                # Chunking would be needed here for N > 5k-10k.
                diff = x.unsqueeze(0) - x.unsqueeze(1) # (N, N, D)
                diff = self._minimum_image(diff)
                dists = diff.norm(dim=-1)
                
            # Mask for cutoff + skin
            # Exclude self-interaction (diagonal) -> dists > 0
            mask = (dists < self.r_cut_skin) & (dists > 1e-6)
            
            # Only keep upper triangular to avoid double counting
            triu_mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
            final_mask = mask & triu_mask
            
            # Get indices
            idx_i, idx_j = final_mask.nonzero(as_tuple=True)
            
            self.neighbor_list_i = idx_i
            self.neighbor_list_j = idx_j
            
            # Save x for displacement check
            if self.last_update_x.shape != x.shape:
                 self.last_update_x = x.detach().clone()
            else:
                 self.last_update_x.copy_(x.detach())
                 
            return len(idx_i)
        
    def check_neighbor_list(self, x: torch.Tensor) -> bool:
        """Check if neighbor list needs updating based on displacement criterion.
        
        Returns True if max displacement > skin/2.
        """
        with torch.no_grad():
            if self.last_update_x.numel() <= 1: 
                 return True
            if x.shape != self.last_update_x.shape:
                 return True
                 
            diff = x - self.last_update_x
            diff = self._minimum_image(diff)
            max_disp = diff.norm(dim=-1).max()
            
            return max_disp > (self.skin * 0.5)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy using stored neighbor list. O(N_neighbors)."""
        # Automatic update if needed (single system only)
        if x.ndim == self.event_dim:
            if self.check_neighbor_list(x):
                self.update_neighbor_list(x)
        elif x.ndim > self.event_dim:
            # Fallback to O(N^2) base implementation for batches
            return super().energy(x)

        if self.neighbor_list_i.numel() == 0:
            # Fallback for empty neighbor list: return 0 connected to x for gradients
            return (x * 0.0).sum()
            
        idx_i = self.neighbor_list_i
        idx_j = self.neighbor_list_j
        
        # Gather positions
        xi = x[idx_i] # (n_pairs, D)
        xj = x[idx_j]
        
        diff = xi - xj
        diff = self._minimum_image(diff)
        r2 = (diff**2).sum(-1)
        
        # Check cutoff (actual potential cutoff, not skin)
        # We can compute LJ for everything in skin (soft) or strict cutoff.
        # Usually strict cutoff.
        r_cut_sq = self.cutoff**2
        
        # Soft mask or conditional? 
        # For full differentiability, we calculate all in list. 
        # Typically we multiply by a switching function or just let it be (tail correction).
        # Here we'll compute for all in list but maybe zero out > cutoff?
        # A discontinuous cutoff introduces force discontinuities.
        # Standard LJ is usually shifted or switched.
        # We will compute for all pairs in the neighbor list (cutoff + skin).
        # Note: This means the potential is slightly different (includes skin interactions)
        # unless we strictly mask.
        
        # Let's apply strict mask for correctness with base LJ?
        # mask_strict = r2 < r_cut_sq
        # r2 = r2[mask_strict]
        # But masking changes shape, breaks JIT/vmap potentially? 
        # In eager mode it's fine.
        
        s2 = self.sigma**2 / r2
        s6 = s2**3
        u_pairs = 4 * self.eps * (s6**2 - s6)
        
        # Optional: Zero out energy beyond cutoff if desired. 
        # u_pairs = torch.where(r2 < r_cut_sq, u_pairs, torch.zeros_like(u_pairs))
        
        return u_pairs.sum()


class Harmonic(Potential):
    """Simple harmonic oscillator: U(x) = 0.5 * k * ||x - x0||².
    
    Input shape: (..., d). Output shape: (...,).
    
    Args:
        k: Spring constant. Differentiable parameter.
        center: Equilibrium position. Differentiable parameter if provided.
    """
    
    def __init__(self, k: float = 1.0, center: torch.Tensor | None = None):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(k))
        if center is not None:
            self.center = nn.Parameter(center.clone())
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
    
    # Plotting style (Nord-inspired, editorial)
    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": ["JetBrains Mono", "DejaVu Sans Mono", "Menlo", "Monaco"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 8.0,
        "axes.labelpad": 5.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": '0.9',
        "figure.facecolor": "#FAFBFC",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FAFBFC",
        "lines.linewidth": 2.0,
    })
    
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.patch.set_facecolor('#FAFBFC')
    
    # Common colormap and style
    cmap = 'BuPu'
    particle_color = '#5E81AC'  # Steel blue from Nord palette
    
    # 1. Double Well 2D
    ax = axes[0, 0]
    dw = DoubleWell2D()
    x = torch.linspace(-2.5, 2.5, 100)
    y = torch.linspace(-2.0, 2.0, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X, Y], dim=-1)
    U = dw.energy(xy)
    
    # Levels for DoubleWell2D
    levels = np.linspace(0, 5, 21)
    U_clipped = torch.clamp(U, max=5.0)
    
    cs = ax.contourf(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, cmap=cmap, extend='max')
    ax.contour(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, colors='k', linewidths=0.3, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Double Well 2D', fontweight='bold')
    plt.colorbar(cs, ax=ax, label='U')
    ax.set_axisbelow(True)
    
    # 2. Müller-Brown contour
    ax = axes[0, 1]
    mb = MullerBrown()
    x = torch.linspace(-1.5, 1.2, 100)
    y = torch.linspace(-0.5, 2.0, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X, Y], dim=-1)
    U = mb.energy(xy)
    
    # Levels for DoubleWell2D (fixed comment)
    levels = np.linspace(-150, 100, 26)
    U_clipped = torch.clamp(U, max=100.0)
    
    cs = ax.contourf(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, cmap=cmap, extend='max')
    ax.contour(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, colors='k', linewidths=0.3, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Müller-Brown', fontweight='bold')
    plt.colorbar(cs, ax=ax, label='U')
    ax.set_axisbelow(True)
    
    # 3. LJ-7 cluster (no PBC)
    ax = axes[1, 0]
    lj = LennardJones()
    angles = torch.linspace(0, 2*np.pi, 7)[:-1]
    r = 1.12
    positions = torch.zeros(7, 2)
    positions[1:, 0] = r * torch.cos(angles)
    positions[1:, 1] = r * torch.sin(angles)
    
    ax.scatter(positions[:, 0].numpy(), positions[:, 1].numpy(), 
               s=500, c=particle_color, edgecolor='k', lw=1.5, zorder=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(f'LJ-7 cluster (U={lj.energy(positions).item():.2f})', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_axisbelow(True)
    
    # 4. LJ cluster with periodic boundary conditions
    ax = axes[1, 1]
    L = 4.0  # box size
    lj_pbc = LennardJones(box_size=L)
    
    # Create a 3x3 grid of particles in the box
    n_side = 3
    spacing = L / n_side
    pos_pbc = torch.zeros(n_side * n_side, 2)
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            pos_pbc[idx, 0] = (i + 0.5) * spacing
            pos_pbc[idx, 1] = (j + 0.5) * spacing
            idx += 1
    
    # Draw the periodic box
    box = plt.Rectangle((0, 0), L, L, fill=False, edgecolor='#333', lw=2, ls='--')
    ax.add_patch(box)
    
    # Draw ghost images (periodic copies) in faded color
    for dx in [-L, 0, L]:
        for dy in [-L, 0, L]:
            if dx == 0 and dy == 0:
                continue  # skip the main box
            ghost_pos = pos_pbc + torch.tensor([dx, dy])
            ax.scatter(ghost_pos[:, 0].numpy(), ghost_pos[:, 1].numpy(),
                      s=200, c=particle_color, alpha=0.2, edgecolor='none', zorder=5)
    
    # Draw main particles
    ax.scatter(pos_pbc[:, 0].numpy(), pos_pbc[:, 1].numpy(),
               s=400, c=particle_color, edgecolor='k', lw=1.5, zorder=10)
    
    ax.set_xlim(-L*0.5, L*1.5)
    ax.set_ylim(-L*0.5, L*1.5)
    ax.set_aspect('equal')
    ax.set_title(f'LJ-9 with PBC (L={L}, U={lj_pbc.energy(pos_pbc).item():.2f})', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_axisbelow(True)
    
    plt.savefig(os.path.join(assets_dir, "potentials.png"), dpi=150, 
                bbox_inches='tight', facecolor='#FAFBFC')
    print(f"Saved potentials plot to assets/potentials.png")
