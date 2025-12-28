"""Tests for integrators."""

import pytest
import torch
import math
from uni_diffsim.integrators import (
    OverdampedLangevin, BAOAB, VelocityVerlet, NoseHoover, NoseHooverChain, ESH, GLE,
    kinetic_energy, temperature,
)
from uni_diffsim.potentials import DoubleWell, Harmonic, MullerBrown
from uni_diffsim.device import available_devices


DEVICES = available_devices()


class TestOverdampedLangevin:
    """Tests for overdamped Langevin integrator."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        integrator = OverdampedLangevin(gamma=1.0, kT=1.0)
        x = torch.randn(10, device=device)
        force_fn = lambda x: -x
        x_new = integrator.step(x, force_fn, dt=0.01)
        assert x_new.shape == x.shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_dimensions(self, device):
        """Should handle arbitrary batch dimensions."""
        integrator = OverdampedLangevin()
        force_fn = lambda x: -x
        for shape in [(10,), (5, 10), (2, 3, 4)]:
            x = torch.randn(shape, device=device)
            x_new = integrator.step(x, force_fn, dt=0.01)
            assert x_new.shape == shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_run_trajectory_shape(self, device):
        """Run should return correct trajectory shape."""
        integrator = OverdampedLangevin()
        x0 = torch.randn(10, device=device)
        force_fn = lambda x: -x
        traj = integrator.run(x0, force_fn, dt=0.01, n_steps=100, store_every=10)
        assert traj.shape == (11, 10)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_drift_toward_minimum(self, device):
        """Particles should drift toward potential minimum on average."""
        torch.manual_seed(42)
        integrator = OverdampedLangevin(gamma=1.0, kT=0.1)
        x0 = torch.full((100,), 2.0, device=device)
        force_fn = lambda x: -x
        traj = integrator.run(x0, force_fn, dt=0.01, n_steps=1000)
        assert traj[-1].abs().mean() < x0.abs().mean()


class TestBAOAB:
    """Tests for BAOAB integrator."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        integrator = BAOAB(gamma=1.0, kT=1.0, mass=1.0)
        x = torch.randn(10, 3, device=device)
        v = torch.randn(10, 3, device=device)
        force_fn = lambda x: -x
        x_new, v_new = integrator.step(x, v, force_fn, dt=0.01)
        assert x_new.shape == x.shape
        assert v_new.shape == v.shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_auto_velocity_init(self, device):
        """Should initialize velocities if not provided."""
        integrator = BAOAB(gamma=1.0, kT=1.0, mass=1.0)
        x0 = torch.randn(10, 3, device=device)
        force_fn = lambda x: -x
        traj_x, traj_v = integrator.run(x0, None, force_fn, dt=0.01, n_steps=10)
        assert traj_x.shape == (11, 10, 3)
        assert traj_v.shape == (11, 10, 3)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_temperature_equilibration(self, device):
        """Temperature should equilibrate to target kT."""
        torch.manual_seed(42)
        target_kT = 1.0
        integrator = BAOAB(gamma=1.0, kT=target_kT, mass=1.0)
        x0 = torch.zeros(100, 3, device=device)
        v0 = torch.randn(100, 3, device=device) * 0.1
        force_fn = lambda x: -x
        traj_x, traj_v = integrator.run(x0, v0, force_fn, dt=0.01, n_steps=2000)
        T_final = temperature(traj_v[-1], mass=1.0).mean()
        assert 0.7 * target_kT < T_final < 1.3 * target_kT


class TestVelocityVerlet:
    """Tests for symplectic Verlet integrator."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        integrator = VelocityVerlet(mass=1.0)
        x = torch.randn(10, 3, device=device)
        v = torch.randn(10, 3, device=device)
        force_fn = lambda x: -x
        x_new, v_new = integrator.step(x, v, force_fn, dt=0.01)
        assert x_new.shape == x.shape
        assert v_new.shape == v.shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_energy_conservation_harmonic(self, device):
        """Energy should be conserved for harmonic oscillator."""
        integrator = VelocityVerlet(mass=1.0)
        harm = Harmonic(k=1.0).to(device)
        x0 = torch.tensor([[1.0, 0.0]], device=device)
        v0 = torch.tensor([[0.0, 1.0]], device=device)
        traj_x, traj_v = integrator.run(x0, v0, harm.force, dt=0.01, n_steps=1000)
        E = harm.energy(traj_x.squeeze(1)) + kinetic_energy(traj_v.squeeze(1))
        E0 = E[0].item()
        rel_error = ((E - E0).abs() / E0).max()
        assert rel_error < 0.01
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_symplecticity_area_preservation(self, device):
        """Phase space area should be preserved."""
        integrator = VelocityVerlet(mass=1.0)
        force_fn = lambda x: -x
        dt = 0.1
        eps = 0.01
        x_base = torch.tensor([[1.0]], device=device)
        v_base = torch.tensor([[0.0]], device=device)
        x0 = torch.cat([x_base, x_base + eps, x_base + eps, x_base])
        v0 = torch.cat([v_base, v_base, v_base + eps, v_base + eps])
        for _ in range(10):
            x0, v0 = integrator.step(x0, v0, force_fn, dt)
        dx1 = (x0[1] - x0[0]).item()
        dv1 = (v0[1] - v0[0]).item()
        dx2 = (x0[3] - x0[0]).item()
        dv2 = (v0[3] - v0[0]).item()
        area_final = abs(dx1 * dv2 - dx2 * dv1)
        area_initial = eps * eps
        assert abs(area_final - area_initial) / area_initial < 0.05


class TestNoseHoover:
    """Tests for single Nosé-Hoover thermostat."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        integrator = NoseHoover(kT=1.0, mass=1.0, Q=1.0)
        x = torch.randn(10, 3, device=device)
        v = torch.randn(10, 3, device=device)
        alpha = torch.zeros(10, device=device)
        force_fn = lambda x: -x
        x_new, v_new, alpha_new = integrator.step(x, v, alpha, force_fn, dt=0.01)
        assert x_new.shape == x.shape
        assert v_new.shape == v.shape
        assert alpha_new.shape == alpha.shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_run_shape(self, device):
        """Run should return correct trajectory shape."""
        integrator = NoseHoover(kT=1.0)
        x0 = torch.randn(10, 3, device=device)
        force_fn = lambda x: -x
        traj_x, traj_v = integrator.run(x0, None, force_fn, dt=0.01, n_steps=100)
        assert traj_x.shape == (101, 10, 3)
        assert traj_v.shape == (101, 10, 3)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_temperature_equilibration(self, device):
        """Temperature should equilibrate to target kT."""
        torch.manual_seed(42)
        target_kT = 1.0
        integrator = NoseHoover(kT=target_kT, mass=1.0, Q=1.0)
        x0 = torch.zeros(100, 2, device=device)
        v0 = torch.randn(100, 2, device=device) * 0.5
        harm = Harmonic(k=1.0).to(device)
        traj_x, traj_v = integrator.run(x0, v0, harm.force, dt=0.01, n_steps=5000)
        # Check temperature in second half of trajectory
        T_samples = temperature(traj_v[2500:], mass=1.0)
        T_mean = T_samples.mean()
        assert 0.5 * target_kT < T_mean < 1.5 * target_kT


class TestNoseHooverChain:
    """Tests for Nosé-Hoover chain thermostat."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        integrator = NoseHooverChain(kT=1.0, mass=1.0, Q=1.0, n_chain=2)
        x = torch.randn(10, 3, device=device)
        v = torch.randn(10, 3, device=device)
        xi = torch.zeros(10, 2, device=device)
        force_fn = lambda x: -x
        x_new, v_new, xi_new = integrator.step(x, v, xi, force_fn, dt=0.01, ndof=3)
        assert x_new.shape == x.shape
        assert v_new.shape == v.shape
        assert xi_new.shape == xi.shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_run_shape(self, device):
        """Run should return correct trajectory shape."""
        integrator = NoseHooverChain(kT=1.0)
        x0 = torch.randn(10, 3, device=device)
        force_fn = lambda x: -x
        traj_x, traj_v = integrator.run(x0, None, force_fn, dt=0.01, n_steps=100)
        assert traj_x.shape == (101, 10, 3)
        assert traj_v.shape == (101, 10, 3)


class TestESH:
    """Tests for Energy Sampling Hamiltonian dynamics."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        esh = ESH(eps=0.1)
        x = torch.randn(10, 2, device=device)
        u = torch.randn(10, 2, device=device)
        u = u / u.norm(dim=-1, keepdim=True)
        r = torch.zeros(10, device=device)
        grad_fn = lambda x: x
        x_new, u_new, r_new = esh.step(x, u, r, grad_fn)
        assert x_new.shape == x.shape
        assert u_new.shape == u.shape
        assert r_new.shape == r.shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_unit_velocity_preserved(self, device):
        """Unit velocity should remain normalized."""
        esh = ESH(eps=0.1)
        x = torch.randn(10, 3, device=device)
        u = torch.randn(10, 3, device=device)
        u = u / u.norm(dim=-1, keepdim=True)
        r = torch.zeros(10, device=device)
        grad_fn = lambda x: x
        for _ in range(10):
            x, u, r = esh.step(x, u, r, grad_fn)
        u_norms = u.norm(dim=-1)
        assert torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_run_shape(self, device):
        """Run should return correct trajectory shape."""
        esh = ESH(eps=0.1)
        x0 = torch.randn(10, 3, device=device)
        grad_fn = lambda x: x
        traj_x, traj_u, traj_r = esh.run(x0, None, grad_fn, n_steps=100, store_every=10)
        assert traj_x.shape == (11, 10, 3)
        assert traj_u.shape == (11, 10, 3)
        assert traj_r.shape == (11, 10)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_deterministic(self, device):
        """ESH should be deterministic (no noise)."""
        esh = ESH(eps=0.1)
        x0 = torch.tensor([[1.0, 0.0]], device=device)
        u0 = torch.tensor([[0.0, 1.0]], device=device)
        grad_fn = lambda x: x
        traj1_x, _, _ = esh.run(x0, u0, grad_fn, n_steps=50)
        traj2_x, _, _ = esh.run(x0, u0, grad_fn, n_steps=50)
        assert torch.allclose(traj1_x, traj2_x)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_explores_space(self, device):
        """ESH should explore different regions of space."""
        esh = ESH(eps=0.5)
        x0 = torch.tensor([[0.0, 0.5]], device=device)
        u0 = torch.tensor([[1.0, 0.0]], device=device)
        mb = MullerBrown().to(device)
        def grad_fn(x):
            return -mb.force(x)
        traj_x, _, _ = esh.run(x0, u0, grad_fn, n_steps=500, store_every=10)
        x_range = traj_x[:, 0, 0].max() - traj_x[:, 0, 0].min()
        assert x_range > 0.3


class TestGLE:
    """Tests for Generalized Langevin Equation integrator."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        gle = GLE(kT=1.0, mass=1.0, gamma=[1.0, 2.0], c=[1.0, 2.0])
        x = torch.randn(10, 3, device=device)
        v = torch.randn(10, 3, device=device)
        s = torch.randn(10, 3, 2, device=device)
        force_fn = lambda x: -x
        x_new, v_new, s_new = gle.step(x, v, s, force_fn, dt=0.01)
        assert x_new.shape == x.shape
        assert v_new.shape == v.shape
        assert s_new.shape == s.shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_run_shape(self, device):
        """Run should return correct trajectory shape."""
        gle = GLE(kT=1.0)
        x0 = torch.randn(10, 3, device=device)
        force_fn = lambda x: -x
        traj_x, traj_v = gle.run(x0, None, force_fn, dt=0.01, n_steps=100, store_every=10)
        assert traj_x.shape == (11, 10, 3)
        assert traj_v.shape == (11, 10, 3)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_single_mode_like_langevin(self, device):
        """Single-mode GLE should behave like standard Langevin."""
        torch.manual_seed(42)
        gle = GLE(kT=1.0, mass=1.0, gamma=[1.0], c=[1.0])
        x0 = torch.zeros(100, 1, device=device)
        force_fn = lambda x: -x
        traj_x, traj_v = gle.run(x0, None, force_fn, dt=0.01, n_steps=2000)
        T_final = temperature(traj_v[-1], mass=1.0).mean()
        assert 0.5 < T_final < 1.5
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_multi_mode(self, device):
        """Multi-mode GLE should run without errors."""
        gle = GLE(kT=1.0, mass=1.0, gamma=[0.5, 2.0, 5.0], c=[0.5, 1.0, 1.5])
        x0 = torch.randn(10, 2, device=device)
        force_fn = lambda x: -x
        traj_x, traj_v = gle.run(x0, None, force_fn, dt=0.01, n_steps=100)
        assert torch.isfinite(traj_x).all()
        assert torch.isfinite(traj_v).all()


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_kinetic_energy(self, device):
        """KE = 0.5 * m * v^2."""
        v = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        KE = kinetic_energy(v, mass=2.0)
        assert torch.isclose(KE, torch.tensor([1.0], device=device))
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_kinetic_energy_batch(self, device):
        """KE should handle batches."""
        v = torch.randn(10, 5, 3, device=device)
        KE = kinetic_energy(v, mass=1.0)
        assert KE.shape == (10, 5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_temperature(self, device):
        """T = 2*KE / ndof."""
        v = torch.tensor([[1.0]], device=device)
        T = temperature(v, mass=1.0)
        assert torch.isclose(T, torch.tensor([1.0], device=device))


class TestVectorization:
    """Test that all integrators are properly vectorized."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_overdamped_batch(self, device):
        """OverdampedLangevin should handle batch dimensions."""
        integrator = OverdampedLangevin(gamma=1.0, kT=1.0)
        force_fn = lambda x: -x
        for shape in [(10, 3), (5, 10, 3), (2, 3, 4, 2)]:
            x = torch.randn(shape, device=device)
            x_new = integrator.step(x, force_fn, dt=0.01)
            assert x_new.shape == shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_baoab_batch(self, device):
        """BAOAB should handle batch dimensions."""
        integrator = BAOAB(gamma=1.0, kT=1.0, mass=1.0)
        force_fn = lambda x: -x
        for shape in [(10, 3), (5, 10, 3), (2, 3, 4, 2)]:
            x = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)
            x_new, v_new = integrator.step(x, v, force_fn, dt=0.01)
            assert x_new.shape == shape
            assert v_new.shape == shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_verlet_batch(self, device):
        """VelocityVerlet should handle batch dimensions."""
        integrator = VelocityVerlet(mass=1.0)
        force_fn = lambda x: -x
        for shape in [(10, 3), (5, 10, 3), (2, 3, 4, 2)]:
            x = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)
            x_new, v_new = integrator.step(x, v, force_fn, dt=0.01)
            assert x_new.shape == shape
            assert v_new.shape == shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_nh_batch(self, device):
        """NoseHoover should handle batch dimensions."""
        integrator = NoseHoover(kT=1.0, mass=1.0, Q=1.0)
        force_fn = lambda x: -x
        for batch_shape, dim in [((10,), 3), ((5, 10), 3), ((2, 3, 4), 2)]:
            shape = batch_shape + (dim,)
            x = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)
            alpha = torch.zeros(batch_shape, device=device)
            x_new, v_new, alpha_new = integrator.step(x, v, alpha, force_fn, dt=0.01)
            assert x_new.shape == shape
            assert v_new.shape == shape
            assert alpha_new.shape == batch_shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_nhc_batch(self, device):
        """NoseHooverChain should handle batch dimensions."""
        integrator = NoseHooverChain(kT=1.0, mass=1.0, Q=1.0, n_chain=2)
        force_fn = lambda x: -x
        for batch_shape, dim in [((10,), 3), ((5, 10), 3), ((2, 3, 4), 2)]:
            shape = batch_shape + (dim,)
            x = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)
            xi = torch.zeros(batch_shape + (2,), device=device)
            x_new, v_new, xi_new = integrator.step(x, v, xi, force_fn, dt=0.01, ndof=dim)
            assert x_new.shape == shape
            assert v_new.shape == shape
            assert xi_new.shape == batch_shape + (2,)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_esh_batch(self, device):
        """ESH should handle batch dimensions."""
        esh = ESH(eps=0.1)
        grad_fn = lambda x: x
        for batch_shape, dim in [((10,), 3), ((5, 10), 3), ((2, 3, 4), 2)]:
            shape = batch_shape + (dim,)
            x = torch.randn(shape, device=device)
            u = torch.randn(shape, device=device)
            u = u / u.norm(dim=-1, keepdim=True)
            r = torch.zeros(batch_shape, device=device)
            x_new, u_new, r_new = esh.step(x, u, r, grad_fn)
            assert x_new.shape == shape
            assert u_new.shape == shape
            assert r_new.shape == batch_shape
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_gle_batch(self, device):
        """GLE should handle batch dimensions."""
        gle = GLE(kT=1.0, mass=1.0, gamma=[1.0, 2.0], c=[1.0, 2.0])
        force_fn = lambda x: -x
        for batch_shape, dim in [((10,), 3), ((5, 10), 3), ((2, 3, 4), 2)]:
            shape = batch_shape + (dim,)
            x = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)
            s = torch.randn(shape + (2,), device=device)
            x_new, v_new, s_new = gle.step(x, v, s, force_fn, dt=0.01)
            assert x_new.shape == shape
            assert v_new.shape == shape
            assert s_new.shape == shape + (2,)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_vectorized_consistency(self, device):
        """Batched computation should match individual computation."""
        torch.manual_seed(42)
        integrator = VelocityVerlet(mass=1.0)
        force_fn = lambda x: -x
        
        # Single trajectory
        x1 = torch.tensor([[1.0, 0.0]], device=device)
        v1 = torch.tensor([[0.0, 1.0]], device=device)
        x1_new, v1_new = integrator.step(x1, v1, force_fn, dt=0.1)
        
        # Batched (same initial conditions in batch)
        x_batch = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device)
        v_batch = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device=device)
        x_batch_new, v_batch_new = integrator.step(x_batch, v_batch, force_fn, dt=0.1)
        
        assert torch.allclose(x1_new, x_batch_new[0:1])
        assert torch.allclose(v1_new, v_batch_new[0:1])
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_vectorization_speedup(self, device):
        """Batched should be faster than sequential (verifies broadcasting)."""
        import time
        
        def sync(device):
            if 'cuda' in str(device) and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif 'mps' in str(device) and torch.backends.mps.is_available():
                torch.mps.synchronize()
        
        integrator = BAOAB(gamma=1.0, kT=1.0, mass=1.0)
        force_fn = lambda x: -x
        n_particles = 100
        dim = 10
        n_steps = 50
        
        # Sequential (loop over particles)
        x_seq = [torch.randn(1, dim, device=device) for _ in range(n_particles)]
        v_seq = [torch.randn(1, dim, device=device) for _ in range(n_particles)]
        sync(device)
        start = time.perf_counter()
        for _ in range(n_steps):
            for i in range(n_particles):
                x_seq[i], v_seq[i] = integrator.step(x_seq[i], v_seq[i], force_fn, dt=0.01)
        sync(device)
        time_seq = time.perf_counter() - start
        
        # Batched (single call)
        x_batch = torch.randn(n_particles, dim, device=device)
        v_batch = torch.randn(n_particles, dim, device=device)
        sync(device)
        start = time.perf_counter()
        for _ in range(n_steps):
            x_batch, v_batch = integrator.step(x_batch, v_batch, force_fn, dt=0.01)
        sync(device)
        time_batch = time.perf_counter() - start
        
        # Batched should be significantly faster
        assert time_batch < time_seq, f"Batched ({time_batch:.4f}s) should be faster than sequential ({time_seq:.4f}s)"


class TestDifferentiability:
    """Test that integrators support gradient computation."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_overdamped_differentiable(self, device):
        """Overdamped Langevin should be differentiable."""
        torch.manual_seed(42)
        integrator = OverdampedLangevin(gamma=1.0, kT=0.1)
        k = torch.tensor([1.0], device=device, requires_grad=True)
        x0 = torch.tensor([1.0], device=device)
        def force_fn(x):
            return -k * x
        x = x0
        for _ in range(10):
            x = integrator.step(x, force_fn, dt=0.01)
        loss = x.pow(2).sum()
        loss.backward()
        assert k.grad is not None
        assert torch.isfinite(k.grad)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_baoab_differentiable(self, device):
        """BAOAB should be differentiable."""
        torch.manual_seed(42)
        integrator = BAOAB(gamma=1.0, kT=0.1, mass=1.0)
        k = torch.tensor([1.0], device=device, requires_grad=True)
        x = torch.tensor([1.0], device=device)
        v = torch.tensor([0.0], device=device)
        def force_fn(x):
            return -k * x
        for _ in range(10):
            x, v = integrator.step(x, v, force_fn, dt=0.01)
        loss = x.pow(2).sum()
        loss.backward()
        assert k.grad is not None
        assert torch.isfinite(k.grad)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_verlet_differentiable(self, device):
        """Verlet should be differentiable."""
        integrator = VelocityVerlet(mass=1.0)
        k = torch.tensor([1.0], device=device, requires_grad=True)
        x = torch.tensor([1.0], device=device)
        v = torch.tensor([0.0], device=device)
        def force_fn(x):
            return -k * x
        for _ in range(10):
            x, v = integrator.step(x, v, force_fn, dt=0.01)
        loss = x.pow(2).sum()
        loss.backward()
        assert k.grad is not None
        assert torch.isfinite(k.grad)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_nh_differentiable(self, device):
        """NoseHoover should be differentiable."""
        nh = NoseHoover(kT=0.5, mass=1.0, Q=1.0)
        k = torch.tensor([1.0], device=device, requires_grad=True)
        x = torch.tensor([[1.0, 0.0]], device=device)
        v = torch.tensor([[0.0, 1.0]], device=device)
        alpha = torch.zeros(1, device=device)
        def force_fn(x):
            return -k * x
        for _ in range(10):
            x, v, alpha = nh.step(x, v, alpha, force_fn, dt=0.01)
        loss = x.pow(2).sum()
        loss.backward()
        assert k.grad is not None
        assert torch.isfinite(k.grad)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_esh_differentiable(self, device):
        """ESH should be differentiable (deterministic)."""
        esh = ESH(eps=0.1)
        k = torch.tensor([1.0], device=device, requires_grad=True)
        x = torch.tensor([[1.0, 0.0]], device=device)
        u = torch.tensor([[0.0, 1.0]], device=device)
        r = torch.zeros(1, device=device)
        def grad_fn(x):
            return k * x
        for _ in range(10):
            x, u, r = esh.step(x, u, r, grad_fn)
        loss = x.pow(2).sum()
        loss.backward()
        assert k.grad is not None
        assert torch.isfinite(k.grad)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_gle_differentiable(self, device):
        """GLE should be differentiable."""
        torch.manual_seed(42)
        gle = GLE(kT=0.1, mass=1.0)
        k = torch.tensor([1.0], device=device, requires_grad=True)
        x = torch.tensor([[1.0]], device=device)
        v = torch.tensor([[0.0]], device=device)
        s = torch.zeros(1, 1, 1, device=device)
        def force_fn(x):
            return -k * x
        for _ in range(10):
            x, v, s = gle.step(x, v, s, force_fn, dt=0.01)
        loss = x.pow(2).sum()
        loss.backward()
        assert k.grad is not None
        assert torch.isfinite(k.grad)
