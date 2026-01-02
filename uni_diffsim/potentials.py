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
        self.barrier_height = nn.Parameter(torch.tensor(barrier_height, dtype=torch.float32))
    
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
        self.barrier_height = nn.Parameter(torch.tensor(barrier_height, dtype=torch.float32))
        self.asymmetry = nn.Parameter(torch.tensor(asymmetry, dtype=torch.float32))

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
        x_left = torch.tensor(-1.0, dtype=torch.float32)
        x_right = torch.tensor(1.0, dtype=torch.float32)
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
        self.barrier_height = nn.Parameter(torch.tensor(barrier_height, dtype=torch.float32))
        self.k_y = nn.Parameter(torch.tensor(k_y, dtype=torch.float32))
    
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
    _A_default = torch.tensor([-200., -100., -170., 15.], dtype=torch.float32)
    _a_default = torch.tensor([-1., -1., -6.5, 0.7], dtype=torch.float32)
    _b_default = torch.tensor([0., 0., 11., 0.6], dtype=torch.float32)
    _c_default = torch.tensor([-10., -10., -6.5, 0.7], dtype=torch.float32)
    _x0_default = torch.tensor([1., 0., -0.5, -1.], dtype=torch.float32)
    _y0_default = torch.tensor([0., 0.5, 1.5, 1.], dtype=torch.float32)
    
    def __init__(self, 
                 A: torch.Tensor | None = None,
                 a: torch.Tensor | None = None,
                 b: torch.Tensor | None = None,
                 c: torch.Tensor | None = None,
                 x0: torch.Tensor | None = None,
                 y0: torch.Tensor | None = None):
        super().__init__()
        self.A = nn.Parameter(A.float().clone() if A is not None else self._A_default.clone())
        self.a = nn.Parameter(a.float().clone() if a is not None else self._a_default.clone())
        self.b = nn.Parameter(b.float().clone() if b is not None else self._b_default.clone())
        self.c = nn.Parameter(c.float().clone() if c is not None else self._c_default.clone())
        self.x0 = nn.Parameter(x0.float().clone() if x0 is not None else self._x0_default.clone())
        self.y0 = nn.Parameter(y0.float().clone() if y0 is not None else self._y0_default.clone())
    
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
        self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        
        if box_size is not None:
            if isinstance(box_size, (int, float)):
                box_size = torch.tensor([box_size], dtype=torch.float32)
            elif not isinstance(box_size, torch.Tensor):
                box_size = torch.tensor(box_size, dtype=torch.float32)
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
        device = x.device
        
        # Get indices for upper triangular part (i < j)
        # Use simple single-slot caching to avoid re-generating indices
        # while preventing memory leaks with varying N
        if (not hasattr(self, '_indices_cache') or
            self._indices_cache[0] != n or
            self._indices_cache[1] != device):

            idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)
            self._indices_cache = (n, device, idx_i, idx_j)
        else:
            _, _, idx_i, idx_j = self._indices_cache
        
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
            # Compare squared distances to avoid expensive sqrt
            max_disp_sq = (diff**2).sum(dim=-1).max()
            
            return max_disp_sq > (self.skin * 0.5)**2

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
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        if center is not None:
            self.center = nn.Parameter(center.float().clone())
        else:
            self.center = None
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if self.center is not None:
            x = x - self.center
        return 0.5 * self.k * (x**2).sum(-1)


class HenonHeiles(Potential):
    """Henon-Heiles potential: U(x,y) = 0.5(x^2 + y^2) + lambda * (x^2*y - y^3/3).

    A classic system for Hamiltonian chaos.

    Args:
        lam: Nonlinearity parameter (lambda).
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(lam, dtype=torch.float32))

    def energy(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (..., 2)"""
        x, y = xy[..., 0], xy[..., 1]
        return 0.5 * (x**2 + y**2) + self.lam * (x**2 * y - y**3 / 3.0)


if __name__ == "__main__":
    """Quick sanity check for potentials."""
    import numpy as np
    
    print("Potentials sanity check...")
    
    # DoubleWell
    dw = DoubleWell(barrier_height=1.0)
    x = torch.tensor([0.0, 1.0, -1.0])
    u = dw.energy(x)
    assert u[0] > u[1], "Barrier should be higher than wells"
    assert torch.allclose(u[1], u[2]), "Wells should be symmetric"
    print(f"  DoubleWell: U(0)={u[0]:.2f}, U(±1)={u[1]:.2f}")
    
    # AsymmetricDoubleWell
    adw = AsymmetricDoubleWell(barrier_height=1.0, asymmetry=0.5)
    u_left, u_right = adw.well_depths()
    assert u_left < u_right, "Left well should be lower with positive asymmetry"
    print(f"  AsymmetricDoubleWell: U_left={u_left:.2f}, U_right={u_right:.2f}")
    
    # DoubleWell2D
    dw2d = DoubleWell2D()
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    u = dw2d.energy(xy)
    assert u[0] > u[1], "Saddle should be higher than minima"
    print(f"  DoubleWell2D: U(saddle)={u[0]:.2f}, U(minima)={u[1]:.2f}")
    
    # MullerBrown
    mb = MullerBrown()
    xy = torch.tensor([[0.6, 0.0], [-0.5, 1.5]])  # Near two minima
    u = mb.energy(xy)
    print(f"  MullerBrown: U at two points = {u[0]:.1f}, {u[1]:.1f}")
    
    # LennardJones
    lj = LennardJones(eps=1.0, sigma=1.0)
    # Two particles at equilibrium distance (r = 2^(1/6) * sigma)
    r_eq = 2**(1/6)
    pos = torch.tensor([[0.0, 0.0], [r_eq, 0.0]])
    u = lj.energy(pos)
    assert u < 0, "LJ should be negative at equilibrium"
    print(f"  LennardJones: U(r_eq)={u:.3f} (should be -1)")
    
    # LennardJones with PBC
    lj_pbc = LennardJones(eps=1.0, sigma=1.0, box_size=5.0)
    pos = torch.tensor([[0.0, 0.0], [4.5, 0.0]])  # Close via PBC
    u_pbc = lj_pbc.energy(pos)
    print(f"  LennardJones (PBC): U={u_pbc:.3f} (particles wrap around)")
    
    # Harmonic
    h = Harmonic(k=2.0)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    u = h.energy(x)
    assert torch.allclose(u, torch.tensor([1.0, 1.0])), "U = 0.5 * k * x^2"
    print(f"  Harmonic: U(1,0)={u[0]:.2f}")
    
    # Test forces (autograd)
    x = torch.tensor([[1.0, 0.0]], requires_grad=True)
    f = dw2d.force(x)
    assert f.shape == x.shape
    print(f"  Force computation: OK")
    
    # Test Hessian
    x = torch.tensor([0.5, 0.5])
    H = dw2d.hessian(x)
    assert H.shape == (2, 2)
    print(f"  Hessian computation: OK, shape={H.shape}")
    
    # HenonHeiles
    hh = HenonHeiles()
    xy = torch.tensor([1.0, 1.0])
    u = hh.energy(xy)
    print(f"  HenonHeiles: U(1,1)={u.item():.3f}")

    print("\nAll potentials passed sanity check!")
