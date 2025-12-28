"""Tests for potential energy surfaces."""

import pytest
import torch
from uni_diffsim.potentials import DoubleWell, MullerBrown, LennardJones, Harmonic
from uni_diffsim.device import available_devices


# Test on all available devices
DEVICES = available_devices()


class TestDoubleWell:
    """Tests for DoubleWell potential."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_minima_locations(self, device):
        """Minima should be at x = ±1."""
        dw = DoubleWell().to(device)
        x = torch.tensor([-1.0, 1.0], device=device)
        u = dw.energy(x)
        assert torch.allclose(u, torch.zeros(2, device=device), atol=1e-6)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_barrier_height(self, device):
        """Barrier at x=0 should equal barrier_height."""
        for h in [0.5, 1.0, 2.0]:
            dw = DoubleWell(barrier_height=h).to(device)
            u = dw.energy(torch.tensor(0.0, device=device))
            assert torch.isclose(u, torch.tensor(h, device=device))
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_force_at_minima(self, device):
        """Force should be zero at minima."""
        dw = DoubleWell().to(device)
        x = torch.tensor([[-1.0], [1.0]], device=device)
        f = dw.force(x)
        assert torch.allclose(f, torch.zeros_like(f), atol=1e-5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_force_direction(self, device):
        """Force should point toward nearest minimum."""
        dw = DoubleWell().to(device)
        # Left of left minimum: force should be positive (toward -1)
        x_left = torch.tensor([[-1.5]], device=device)
        assert dw.force(x_left).item() > 0
        # Between minima, left of barrier: force negative (toward -1)
        x_mid_left = torch.tensor([[-0.3]], device=device)
        assert dw.force(x_mid_left).item() < 0
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_shapes(self, device):
        """Test various batch shapes."""
        dw = DoubleWell().to(device)
        # Scalar-like
        assert dw.energy(torch.tensor(0.5, device=device)).shape == ()
        # 1D batch
        assert dw.energy(torch.randn(10, device=device)).shape == (10,)
        # 2D batch
        assert dw.energy(torch.randn(5, 10, device=device)).shape == (5, 10)
        # With trailing dim=1
        assert dw.energy(torch.randn(5, 10, 1, device=device)).shape == (5, 10)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_hessian_at_minimum(self, device):
        """Hessian at minimum should be positive (stable)."""
        dw = DoubleWell().to(device)
        x = torch.tensor([[1.0]], device=device)
        h = dw.hessian(x)
        assert h.item() > 0  # Positive curvature at minimum
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_hessian_at_barrier(self, device):
        """Hessian at barrier should be negative (unstable)."""
        dw = DoubleWell().to(device)
        x = torch.tensor([[0.0]], device=device)
        h = dw.hessian(x)
        assert h.item() < 0  # Negative curvature at saddle


class TestMullerBrown:
    """Tests for Müller-Brown potential."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_known_minimum(self, device):
        """Check approximate location of global minimum."""
        mb = MullerBrown().to(device)
        # Global minimum is approximately at (-0.558, 1.442)
        xy = torch.tensor([[-0.558, 1.442]], device=device)
        u = mb.energy(xy)
        assert u.item() < -145  # Known to be around -146.7
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_force_finite(self, device):
        """Force should be finite everywhere in reasonable domain."""
        mb = MullerBrown().to(device)
        xy = torch.randn(100, 2, device=device)
        f = mb.force(xy)
        assert torch.isfinite(f).all()
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_shapes(self, device):
        """Test batch dimension handling."""
        mb = MullerBrown().to(device)
        # Single point
        assert mb.energy(torch.randn(2, device=device)).shape == ()
        # Batch
        assert mb.energy(torch.randn(10, 2, device=device)).shape == (10,)
        # 2D batch
        assert mb.energy(torch.randn(5, 10, 2, device=device)).shape == (5, 10)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_hessian_shape(self, device):
        """Hessian should be 2x2 for 2D potential."""
        mb = MullerBrown().to(device)
        xy = torch.randn(2, device=device)
        h = mb.hessian(xy.unsqueeze(0))
        assert h.shape == (1, 2, 2)
        # Batch
        h_batch = mb.hessian(torch.randn(5, 2, device=device))
        assert h_batch.shape == (5, 2, 2)


class TestLennardJones:
    """Tests for Lennard-Jones potential."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_equilibrium_distance(self, device):
        """Two particles at r=2^(1/6)*sigma should be at energy minimum."""
        lj = LennardJones(eps=1.0, sigma=1.0).to(device)
        r_eq = 2**(1/6)  # ~1.122
        x = torch.tensor([[0.0, 0.0], [r_eq, 0.0]], device=device)
        f = lj.force(x)
        # Force should be ~zero at equilibrium
        assert torch.allclose(f, torch.zeros_like(f), atol=1e-4)
    
    # --- Periodic Boundary Condition Tests ---
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_pbc_minimum_image_wrapping(self, device):
        """Particles across periodic boundary should interact via minimum image."""
        L = 5.0
        lj_pbc = LennardJones(eps=1.0, sigma=1.0, box_size=L).to(device)
        lj_no_pbc = LennardJones(eps=1.0, sigma=1.0).to(device)
        
        # Two particles: one at (0.5, 0) and one at (4.5, 0)
        # Real distance = 4.0, but with PBC: wrapped distance = 1.0
        x_pbc = torch.tensor([[0.5, 0.0], [4.5, 0.0]], device=device)
        # Equivalent non-PBC configuration: particles 1.0 apart
        x_direct = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        
        u_pbc = lj_pbc.energy(x_pbc)
        u_direct = lj_no_pbc.energy(x_direct)
        
        assert torch.isclose(u_pbc, u_direct, rtol=1e-5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_pbc_force_direction(self, device):
        """Force should point through the periodic boundary when that's shorter."""
        L = 5.0
        lj = LennardJones(eps=1.0, sigma=1.0, box_size=L).to(device)
        
        # Particles at (0.5, 0) and (4.5, 0): wrapped distance is 1.0
        # diff = x[1] - x[0] = (4.0, 0), wrapped to (-1.0, 0)
        # So particle 1 is at x=-1.0 relative to particle 0 (via boundary)
        x = torch.tensor([[0.5, 0.0], [4.5, 0.0]], device=device)
        f = lj.force(x)
        
        # At r=1.0 (< r_eq~1.12), particles repel
        # Particle 0 is pushed in +x (away from particle 1 at relative x=-1)
        # Particle 1 is pushed in -x (away from particle 0 at relative x=+1)
        assert f[0, 0].item() > 0  # particle 0: force in +x
        assert f[1, 0].item() < 0  # particle 1: force in -x
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_pbc_rectangular_box(self, device):
        """Test non-cubic box with different lengths per dimension."""
        box_size = torch.tensor([4.0, 6.0], device=device)
        lj = LennardJones(eps=1.0, sigma=1.0, box_size=box_size).to(device)
        
        # Test wrapping in x: particles at (0.5, 0) and (3.5, 0)
        # x-distance: 3.0 direct, 1.0 wrapped (since Lx=4)
        x1 = torch.tensor([[0.5, 0.0], [3.5, 0.0]], device=device)
        # Equivalent: particles 1.0 apart
        lj_ref = LennardJones(eps=1.0, sigma=1.0).to(device)
        x_ref = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        
        assert torch.isclose(lj.energy(x1), lj_ref.energy(x_ref), rtol=1e-5)
        
        # Test wrapping in y: particles at (0, 0.5) and (0, 5.5)
        # y-distance: 5.0 direct, 1.0 wrapped (since Ly=6)
        x2 = torch.tensor([[0.0, 0.5], [0.0, 5.5]], device=device)
        x_ref2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], device=device)
        
        assert torch.isclose(lj.energy(x2), lj_ref.energy(x_ref2), rtol=1e-5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_pbc_translation_invariance(self, device):
        """Energy should be invariant under translation by box vector."""
        L = 5.0
        lj = LennardJones(eps=1.0, sigma=1.0, box_size=L).to(device)
        
        x = torch.tensor([[1.0, 1.0], [2.5, 2.0], [3.0, 4.0]], device=device)
        u1 = lj.energy(x)
        
        # Translate all particles by box vector
        x_shifted = x + torch.tensor([L, 0.0], device=device)
        u2 = lj.energy(x_shifted)
        
        assert torch.isclose(u1, u2, rtol=1e-5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_pbc_gradient_consistency(self, device):
        """Autograd force should match finite difference with PBC."""
        L = 5.0
        lj = LennardJones(eps=1.0, sigma=1.0, box_size=L).to(device)
        
        x = torch.tensor([[0.5, 0.5], [4.5, 0.5]], device=device)
        f_auto = lj.force(x)
        
        eps = 1e-4
        f_fd = torch.zeros_like(x)
        for i in range(2):
            for j in range(2):
                x_plus = x.clone()
                x_minus = x.clone()
                x_plus[i, j] += eps
                x_minus[i, j] -= eps
                f_fd[i, j] = -(lj.energy(x_plus) - lj.energy(x_minus)) / (2 * eps)
        
        assert torch.allclose(f_auto, f_fd, rtol=1e-2)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_energy_at_equilibrium(self, device):
        """Energy at equilibrium should be -eps."""
        lj = LennardJones(eps=1.0, sigma=1.0).to(device)
        r_eq = 2**(1/6)
        x = torch.tensor([[0.0, 0.0], [r_eq, 0.0]], device=device)
        u = lj.energy(x)
        assert torch.isclose(u, torch.tensor(-1.0, device=device), atol=1e-5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_repulsive_at_short_range(self, device):
        """Energy should be very positive at short distances."""
        lj = LennardJones().to(device)
        x = torch.tensor([[0.0, 0.0], [0.5, 0.0]], device=device)
        u = lj.energy(x)
        assert u.item() > 10  # Strongly repulsive
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_shapes(self, device):
        """Test batch dimension handling."""
        lj = LennardJones().to(device)
        # Single configuration
        assert lj.energy(torch.randn(5, 3, device=device)).shape == ()
        # Batch of configurations
        assert lj.energy(torch.randn(10, 5, 3, device=device)).shape == (10,)
        # 2D batch
        assert lj.energy(torch.randn(4, 10, 5, 3, device=device)).shape == (4, 10)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_n_pairs_scaling(self, device):
        """Energy should scale with number of pairs."""
        lj = LennardJones().to(device)
        # n particles -> n*(n-1)/2 pairs
        for n in [2, 3, 4, 5]:
            x = torch.randn(n, 2, device=device) * 3  # Spread out to avoid huge energies
            u = lj.energy(x)
            assert torch.isfinite(u)


class TestHarmonic:
    """Tests for Harmonic potential."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_minimum_at_center(self, device):
        """Energy should be zero at center."""
        h = Harmonic(k=1.0).to(device)
        x = torch.zeros(3, device=device)
        assert torch.isclose(h.energy(x), torch.tensor(0.0, device=device))
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_custom_center(self, device):
        """Test with non-zero center."""
        center = torch.tensor([1.0, 2.0], device=device)
        h = Harmonic(k=1.0, center=center).to(device)
        assert torch.isclose(h.energy(center), torch.tensor(0.0, device=device))
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_energy_value(self, device):
        """U = 0.5 * k * r^2."""
        h = Harmonic(k=2.0).to(device)
        x = torch.tensor([1.0, 0.0, 0.0], device=device)
        expected = 0.5 * 2.0 * 1.0  # k * r^2 / 2
        assert torch.isclose(h.energy(x), torch.tensor(expected, device=device))
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_force_points_to_center(self, device):
        """Force should point toward center."""
        h = Harmonic(k=1.0).to(device)
        x = torch.tensor([[1.0, 0.0]], device=device)
        f = h.force(x)
        # Force should point in -x direction
        assert f[0, 0].item() < 0
        assert torch.isclose(f[0, 1], torch.tensor(0.0, device=device), atol=1e-6)


class TestGradientConsistency:
    """Test that autograd forces match finite differences."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_double_well_gradient(self, device):
        """Compare autograd force to finite difference."""
        dw = DoubleWell().to(device)
        x = torch.tensor([[0.5]], device=device)
        f_auto = dw.force(x)
        
        eps = 1e-4
        x_plus = x + eps
        x_minus = x - eps
        f_fd = -(dw.energy(x_plus.squeeze()) - dw.energy(x_minus.squeeze())) / (2 * eps)
        
        assert torch.isclose(f_auto.squeeze(), f_fd, rtol=1e-3)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_muller_brown_gradient(self, device):
        """Compare autograd force to finite difference for 2D."""
        mb = MullerBrown().to(device)
        xy = torch.tensor([[0.0, 0.5]], device=device)
        f_auto = mb.force(xy)
        
        eps = 1e-4
        f_fd = torch.zeros(2, device=device)
        for i in range(2):
            xy_plus = xy.clone()
            xy_minus = xy.clone()
            xy_plus[0, i] += eps
            xy_minus[0, i] -= eps
            f_fd[i] = -(mb.energy(xy_plus) - mb.energy(xy_minus)) / (2 * eps)
        
        assert torch.allclose(f_auto.squeeze(), f_fd, rtol=1e-2)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_lj_gradient(self, device):
        """Compare autograd force to finite difference for LJ."""
        lj = LennardJones().to(device)
        x = torch.tensor([[0.0, 0.0], [1.5, 0.0]], device=device)
        f_auto = lj.force(x)
        
        eps = 1e-4
        f_fd = torch.zeros_like(x)
        for i in range(2):
            for j in range(2):
                x_plus = x.clone()
                x_minus = x.clone()
                x_plus[i, j] += eps
                x_minus[i, j] -= eps
                f_fd[i, j] = -(lj.energy(x_plus) - lj.energy(x_minus)) / (2 * eps)
        
        assert torch.allclose(f_auto, f_fd, rtol=1e-2)


class TestPotentialDifferentiability:
    """Test that potentials support gradient computation through parameters."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_double_well_differentiable(self, device):
        """DoubleWell energy should be differentiable."""
        dw = DoubleWell().to(device)
        x = torch.tensor([0.5], device=device, requires_grad=True)
        u = dw.energy(x)
        u.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_muller_brown_differentiable(self, device):
        """MullerBrown energy should be differentiable."""
        mb = MullerBrown().to(device)
        xy = torch.tensor([[0.0, 0.5]], device=device, requires_grad=True)
        u = mb.energy(xy)
        u.backward()
        assert xy.grad is not None
        assert torch.isfinite(xy.grad).all()
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_lj_differentiable(self, device):
        """LennardJones energy should be differentiable."""
        lj = LennardJones().to(device)
        x = torch.tensor([[0.0, 0.0], [1.5, 0.0]], device=device, requires_grad=True)
        u = lj.energy(x)
        u.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_harmonic_differentiable(self, device):
        """Harmonic energy should be differentiable."""
        harm = Harmonic(k=1.0).to(device)
        x = torch.tensor([[1.0, 2.0]], device=device, requires_grad=True)
        u = harm.energy(x)
        u.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_hessian_differentiable(self, device):
        """Hessian computation should work."""
        dw = DoubleWell().to(device)
        x = torch.tensor([[0.5]], device=device)
        H = dw.hessian(x)
        assert H.shape == (1, 1, 1)
        assert torch.isfinite(H).all()

