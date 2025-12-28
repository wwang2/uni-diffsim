"""Tests for integrators."""

import pytest
import torch
import math
from uni_diffsim.integrators import (
    OverdampedLangevin, BAOAB, VelocityVerlet, NoseHooverChain,
    kinetic_energy, temperature,
)
from uni_diffsim.potentials import DoubleWell, Harmonic
from uni_diffsim.device import available_devices


DEVICES = available_devices()


class TestOverdampedLangevin:
    """Tests for overdamped Langevin integrator."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_step_shape(self, device):
        """Step should preserve shape."""
        integrator = OverdampedLangevin(gamma=1.0, kT=1.0)
        x = torch.randn(10, device=device)
        force_fn = lambda x: -x  # Harmonic
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
        assert traj.shape == (11, 10)  # 100/10 + 1 = 11 stored frames
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_drift_toward_minimum(self, device):
        """Particles should drift toward potential minimum on average."""
        torch.manual_seed(42)
        integrator = OverdampedLangevin(gamma=1.0, kT=0.1)
        x0 = torch.full((100,), 2.0, device=device)  # Start far from minimum
        force_fn = lambda x: -x  # Harmonic centered at 0
        traj = integrator.run(x0, force_fn, dt=0.01, n_steps=1000)
        # Final positions should be closer to 0 than initial
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
        v0 = torch.randn(100, 3, device=device) * 0.1  # Start cold
        force_fn = lambda x: -x
        traj_x, traj_v = integrator.run(x0, v0, force_fn, dt=0.01, n_steps=2000)
        
        # Measure temperature from final velocities
        T_final = temperature(traj_v[-1], mass=1.0).mean()
        # Should be within 30% of target (statistical fluctuations)
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
        
        # Energy should be conserved within 1%
        rel_error = ((E - E0).abs() / E0).max()
        assert rel_error < 0.01
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_symplecticity_area_preservation(self, device):
        """Phase space area should be preserved (2D test)."""
        integrator = VelocityVerlet(mass=1.0)
        force_fn = lambda x: -x  # Harmonic
        dt = 0.1
        
        # Create a small parallelogram in phase space
        eps = 0.01
        x_base = torch.tensor([[1.0]], device=device)
        v_base = torch.tensor([[0.0]], device=device)
        
        # Four corners
        x0 = torch.cat([x_base, x_base + eps, x_base + eps, x_base])
        v0 = torch.cat([v_base, v_base, v_base + eps, v_base + eps])
        
        # Evolve
        for _ in range(10):
            x0, v0 = integrator.step(x0, v0, force_fn, dt)
        
        # Compute area (cross product of edge vectors)
        dx1 = (x0[1] - x0[0]).item()
        dv1 = (v0[1] - v0[0]).item()
        dx2 = (x0[3] - x0[0]).item()
        dv2 = (v0[3] - v0[0]).item()
        area_final = abs(dx1 * dv2 - dx2 * dv1)
        area_initial = eps * eps
        
        # Area should be preserved within 5%
        assert abs(area_final - area_initial) / area_initial < 0.05


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


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_kinetic_energy(self, device):
        """KE = 0.5 * m * v^2."""
        v = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        KE = kinetic_energy(v, mass=2.0)
        assert torch.isclose(KE, torch.tensor([1.0], device=device))  # 0.5 * 2 * 1^2
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_kinetic_energy_batch(self, device):
        """KE should handle batches."""
        v = torch.randn(10, 5, 3, device=device)
        KE = kinetic_energy(v, mass=1.0)
        assert KE.shape == (10, 5)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_temperature(self, device):
        """T = 2*KE / ndof."""
        # For 1D with KE = 0.5, T should be 1.0
        v = torch.tensor([[1.0]], device=device)
        T = temperature(v, mass=1.0)
        assert torch.isclose(T, torch.tensor([1.0], device=device))


class TestDifferentiability:
    """Test that integrators support gradient computation."""
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_overdamped_differentiable(self, device):
        """Overdamped Langevin should be differentiable through trajectory."""
        torch.manual_seed(42)
        integrator = OverdampedLangevin(gamma=1.0, kT=0.1)
        
        # Learnable force parameter
        k = torch.tensor([1.0], device=device, requires_grad=True)
        x0 = torch.tensor([1.0], device=device)
        
        def force_fn(x):
            return -k * x
        
        # Run short trajectory
        x = x0
        for _ in range(10):
            x = integrator.step(x, force_fn, dt=0.01)
        
        # Compute loss and gradient
        loss = x.pow(2).sum()
        loss.backward()
        
        assert k.grad is not None
        assert torch.isfinite(k.grad)
    
    @pytest.mark.parametrize("device", DEVICES)
    def test_baoab_differentiable(self, device):
        """BAOAB should be differentiable through trajectory."""
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
        """Verlet should be differentiable (deterministic)."""
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

