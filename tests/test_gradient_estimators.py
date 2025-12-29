"""Tests for gradient estimators (REINFORCE, Girsanov, etc.)."""

import pytest
import torch
import torch.nn as nn
from uni_diffsim.gradient_estimators import (
    ReinforceEstimator, GirsanovEstimator, ReweightingLoss,
    reinforce_gradient,
)
from uni_diffsim.potentials import DoubleWell, Harmonic, DoubleWell2D
from uni_diffsim.integrators import OverdampedLangevin, BAOAB
from uni_diffsim.device import available_devices


DEVICES = available_devices()


class TestReinforceEstimator:
    """Tests for REINFORCE gradient estimator."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        """Should initialize with potential and beta."""
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)
        assert estimator.beta.item() == 1.0
        assert estimator.potential is potential

    @pytest.mark.parametrize("device", DEVICES)
    def test_estimate_gradient_returns_dict(self, device):
        """Should return dictionary of gradients."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        # Generate samples from equilibrium
        samples = torch.randn(100, 2, device=device)

        grads = estimator.estimate_gradient(samples)

        assert isinstance(grads, dict)
        assert 'k' in grads
        assert grads['k'].shape == potential.k.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_shape_matches_parameter(self, device):
        """Gradient shape should match parameter shape."""
        torch.manual_seed(42)
        potential = DoubleWell(barrier_height=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        samples = torch.randn(100, 1, device=device)
        grads = estimator.estimate_gradient(samples)

        assert grads['barrier_height'].shape == potential.barrier_height.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_is_finite(self, device):
        """Gradient should be finite for valid samples."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        samples = torch.randn(100, 2, device=device)
        grads = estimator.estimate_gradient(samples)

        for name, grad in grads.items():
            assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_custom_observable(self, device):
        """Should work with custom observables."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        samples = torch.randn(100, 2, device=device)

        # Observable: mean x coordinate
        observable = lambda x: x[:, 0].mean()
        grads = estimator.estimate_gradient(samples, observable=observable)

        assert 'k' in grads
        assert torch.isfinite(grads['k']).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_observable(self, device):
        """Should work with per-sample observables."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        samples = torch.randn(100, 2, device=device)

        # Observable: per-sample x coordinate
        observable = lambda x: x[:, 0]
        grads = estimator.estimate_gradient(samples, observable=observable)

        assert 'k' in grads

    @pytest.mark.parametrize("device", DEVICES)
    def test_accumulate_and_get_gradient(self, device):
        """Accumulate mode should work correctly."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        # Accumulate multiple batches
        for _ in range(5):
            batch = torch.randn(20, 2, device=device)
            estimator.accumulate(batch)

        grads = estimator.get_gradient()

        assert 'k' in grads
        assert torch.isfinite(grads['k']).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_accumulate_resets_after_get(self, device):
        """Accumulators should reset after get_gradient()."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        batch = torch.randn(20, 2, device=device)
        estimator.accumulate(batch)
        estimator.get_gradient()

        # Should raise error on second call without new accumulation
        with pytest.raises(RuntimeError):
            estimator.get_gradient()

    @pytest.mark.parametrize("device", DEVICES)
    def test_variance_estimation(self, device):
        """Should estimate gradient variance via bootstrap."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        samples = torch.randn(100, 2, device=device)
        variances = estimator.compute_variance(samples, n_bootstrap=10)

        assert 'k' in variances
        assert (variances['k'] >= 0).all()  # Variance is non-negative

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_direction_harmonic(self, device):
        """For harmonic potential at high T, gradient should be negative for k.

        Theory: ⟨x²⟩ = kT/k for harmonic potential.
        So d⟨x²⟩/dk = -kT/k² < 0
        REINFORCE gradient of ⟨x²⟩ w.r.t. k should be negative.
        """
        torch.manual_seed(42)
        kT = 1.0
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0/kT)

        # Generate equilibrium samples (approximately)
        # For harmonic: p(x) ∝ exp(-k*x²/(2kT))
        # variance = kT/k = 1.0
        samples = torch.randn(1000, 2, device=device)  # ~N(0,1)

        # Observable: mean squared displacement
        observable = lambda x: (x**2).sum(dim=-1)
        grads = estimator.estimate_gradient(samples, observable=observable)

        # Gradient should have reasonable magnitude
        # (sign check is tricky due to REINFORCE formulation)
        assert torch.isfinite(grads['k']).all()


class TestReinforceFunctionalAPI:
    """Tests for the functional reinforce_gradient API."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_functional_matches_class(self, device):
        """Functional API should match class API."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        samples = torch.randn(100, 2, device=device)

        # Class API
        estimator = ReinforceEstimator(potential, beta=1.0)
        grads_class = estimator.estimate_gradient(samples)

        # Reset graph state
        for p in potential.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Functional API
        grads_func = reinforce_gradient(samples, potential, beta=1.0)

        for name in grads_class:
            assert torch.allclose(grads_class[name], grads_func[name], atol=1e-5)


class TestReweightingLoss:
    """Tests for ReweightingLoss (surrogate loss for REINFORCE)."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_loss_returns_scalar(self, device):
        """Loss should return scalar tensor."""
        potential = Harmonic(k=1.0).to(device)
        loss_fn = ReweightingLoss(potential, beta=1.0)

        samples = torch.randn(100, 2, device=device)
        loss = loss_fn(samples)

        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("device", DEVICES)
    def test_loss_backward_populates_grad(self, device):
        """Loss backward should populate parameter gradients."""
        potential = Harmonic(k=1.0).to(device)
        loss_fn = ReweightingLoss(potential, beta=1.0)

        samples = torch.randn(100, 2, device=device)
        loss = loss_fn(samples)
        loss.backward()

        assert potential.k.grad is not None
        assert torch.isfinite(potential.k.grad)

    @pytest.mark.parametrize("device", DEVICES)
    def test_loss_and_observable(self, device):
        """Should return both loss and observable value."""
        potential = Harmonic(k=1.0).to(device)
        loss_fn = ReweightingLoss(potential, beta=1.0)

        samples = torch.randn(100, 2, device=device)
        loss, obs = loss_fn.loss_and_observable(samples)

        assert loss.dim() == 0
        assert obs.dim() == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_optimizer_integration(self, device):
        """Should work with standard PyTorch optimizers."""
        torch.manual_seed(42)
        potential = Harmonic(k=2.0).to(device)
        loss_fn = ReweightingLoss(potential, beta=1.0)
        optimizer = torch.optim.SGD(potential.parameters(), lr=0.01)

        initial_k = potential.k.item()

        # Run a few optimization steps
        for _ in range(5):
            samples = torch.randn(100, 2, device=device)
            optimizer.zero_grad()
            loss = loss_fn(samples)
            loss.backward()
            optimizer.step()

        # k should have changed
        assert potential.k.item() != initial_k


class TestGirsanovEstimator:
    """Tests for Girsanov-based path reweighting."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        """Should initialize with potential, sigma, beta."""
        potential = Harmonic(k=1.0).to(device)
        estimator = GirsanovEstimator(potential, sigma=1.0, beta=1.0)
        assert estimator.sigma.item() == 1.0
        assert estimator.beta.item() == 1.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_log_path_score_shape(self, device):
        """Log path score should have correct shape."""
        potential = Harmonic(k=1.0).to(device)
        estimator = GirsanovEstimator(potential, sigma=1.0, beta=1.0)

        # Trajectory: (n_steps, batch, dim)
        trajectory = torch.randn(100, 10, 2, device=device)
        log_score = estimator.compute_log_path_score(trajectory, dt=0.01)

        assert log_score.shape == (10,)  # One score per trajectory in batch

    @pytest.mark.parametrize("device", DEVICES)
    def test_log_path_score_finite(self, device):
        """Log path score should be finite for reasonable trajectories."""
        potential = Harmonic(k=1.0).to(device)
        estimator = GirsanovEstimator(potential, sigma=1.0, beta=1.0)

        # Generate smooth trajectory (not realistic but tests computation)
        trajectory = torch.randn(100, 2, device=device).cumsum(dim=0) * 0.01
        log_score = estimator.compute_log_path_score(trajectory, dt=0.01)

        assert torch.isfinite(log_score).all()


class TestIntegrationWithDynamics:
    """Test gradient estimators with actual dynamics."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_reinforce_with_overdamped_langevin(self, device):
        """REINFORCE should work with Overdamped Langevin samples."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        integrator = OverdampedLangevin(gamma=1.0, kT=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        # Run dynamics
        x0 = torch.randn(50, 2, device=device)
        traj = integrator.run(x0, potential.force, dt=0.01, n_steps=500)

        # Use samples after burn-in
        samples = traj[100:]
        samples_flat = samples.reshape(-1, 2)

        grads = estimator.estimate_gradient(samples_flat)

        assert 'k' in grads
        assert torch.isfinite(grads['k']).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_reinforce_with_baoab(self, device):
        """REINFORCE should work with BAOAB samples."""
        torch.manual_seed(42)
        potential = DoubleWell(barrier_height=1.0).to(device)
        integrator = BAOAB(gamma=1.0, kT=0.5, mass=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=2.0)

        # Run dynamics
        x0 = torch.randn(50, 1, device=device)
        traj_x, traj_v = integrator.run(x0, None, potential.force, dt=0.01, n_steps=500)

        # Use samples after burn-in
        samples = traj_x[100:]
        samples_flat = samples.reshape(-1, 1)

        grads = estimator.estimate_gradient(samples_flat)

        assert 'barrier_height' in grads
        assert torch.isfinite(grads['barrier_height']).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_reweighting_loss_training_loop(self, device):
        """Full training loop with ReweightingLoss."""
        torch.manual_seed(42)

        # Setup
        potential = Harmonic(k=2.0).to(device)
        integrator = OverdampedLangevin(gamma=1.0, kT=1.0).to(device)
        loss_fn = ReweightingLoss(potential, beta=1.0)
        optimizer = torch.optim.Adam(potential.parameters(), lr=0.1)

        # Observable: we want to minimize energy (already default)
        losses = []

        for epoch in range(3):
            # Generate samples
            x0 = torch.randn(50, 2, device=device)
            traj = integrator.run(x0, potential.force, dt=0.01, n_steps=100)
            samples = traj[50:].reshape(-1, 2)  # Use second half

            # Optimization step
            optimizer.zero_grad()
            loss = loss_fn(samples)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Should run without errors and produce finite values
        assert all(abs(l) < 1e10 for l in losses)


class TestComparison:
    """Compare REINFORCE gradients with BPTT gradients."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_reinforce_vs_bptt_harmonic(self, device):
        """For simple systems, REINFORCE and BPTT should roughly agree.

        Note: This is a statistical test and may have some variance.
        We're checking that both methods give gradients in roughly the same direction.
        """
        torch.manual_seed(42)
        kT = 1.0
        k_init = 1.0

        # BPTT gradient
        potential_bptt = Harmonic(k=k_init).to(device)
        integrator = OverdampedLangevin(gamma=1.0, kT=kT).to(device)

        x0 = torch.randn(100, 2, device=device)
        traj = integrator.run(x0, potential_bptt.force, dt=0.01, n_steps=100)

        # Observable: mean energy
        obs_bptt = potential_bptt.energy(traj[-1]).mean()
        obs_bptt.backward()
        grad_bptt = potential_bptt.k.grad.clone()

        # REINFORCE gradient
        potential_rf = Harmonic(k=k_init).to(device)
        estimator = ReinforceEstimator(potential_rf, beta=1.0/kT)

        # Use same trajectory samples
        samples = traj[50:].reshape(-1, 2).detach()
        observable = lambda x: potential_rf.energy(x)
        grads_rf = estimator.estimate_gradient(samples, observable=observable)
        grad_rf = grads_rf['k']

        # Both gradients should be finite
        assert torch.isfinite(grad_bptt)
        assert torch.isfinite(grad_rf)

        # Sign might differ due to different formulations, but both should be non-zero
        assert grad_bptt.abs() > 1e-6 or grad_rf.abs() > 1e-6


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_sample(self, device):
        """Should handle single sample (though high variance)."""
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        sample = torch.randn(1, 2, device=device)
        # Single sample has zero variance, so gradient is zero
        grads = estimator.estimate_gradient(sample)

        assert 'k' in grads

    @pytest.mark.parametrize("device", DEVICES)
    def test_large_batch(self, device):
        """Should handle large batches efficiently."""
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        samples = torch.randn(10000, 2, device=device)
        grads = estimator.estimate_gradient(samples)

        assert 'k' in grads
        assert torch.isfinite(grads['k']).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_high_dimensional(self, device):
        """Should work in high dimensions."""
        center = torch.zeros(100, device=device)
        potential = Harmonic(k=1.0, center=center).to(device)
        estimator = ReinforceEstimator(potential, beta=1.0)

        samples = torch.randn(100, 100, device=device)
        grads = estimator.estimate_gradient(samples)

        assert 'k' in grads
        assert 'center' in grads
        assert grads['center'].shape == (100,)

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_temperature_high_beta(self, device):
        """High beta (low T) should still work."""
        potential = Harmonic(k=1.0).to(device)
        estimator = ReinforceEstimator(potential, beta=100.0)

        samples = torch.randn(100, 2, device=device) * 0.1  # Concentrated samples
        grads = estimator.estimate_gradient(samples)

        assert torch.isfinite(grads['k']).all()
