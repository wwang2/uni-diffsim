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
        with torch.enable_grad():
            if not x.requires_grad:
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
        # Pairwise displacements: (..., n, n, dim)
        diff = x.unsqueeze(-2) - x.unsqueeze(-3)
        
        # Apply minimum image convention for PBC
        diff = self._minimum_image(diff)
        
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
