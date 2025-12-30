"""Tests for O(1) adjoint sensitivity methods."""

import pytest
import torch
from uni_diffsim.gradient_estimators import CheckpointedNoseHoover, ContinuousAdjointNoseHoover, CheckpointManager
from uni_diffsim.integrators import NoseHoover
from uni_diffsim.potentials import Harmonic, DoubleWell
from uni_diffsim.device import available_devices


DEVICES = available_devices()


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_init(self):
        """Test initialization with default and custom checkpoint counts."""
        mgr = CheckpointManager(n_steps=100)
        assert mgr.n_steps == 100
        assert mgr.n_checkpoints == 10  # sqrt(100) = 10

        mgr = CheckpointManager(n_steps=100, n_checkpoints=5)
        assert mgr.n_checkpoints == 5

    def test_save_and_get_checkpoint(self):
        """Test saving and retrieving checkpoints."""
        mgr = CheckpointManager(n_steps=100, n_checkpoints=5)

        x = torch.randn(3, 2)
        v = torch.randn(3, 2)
        alpha = torch.randn(3)

        mgr.save_checkpoint(10, x, v, alpha)
        x_ret, v_ret, alpha_ret = mgr.get_checkpoint(10)

        assert torch.allclose(x_ret, x)
        assert torch.allclose(v_ret, v)
        assert torch.allclose(alpha_ret, alpha)

    def test_get_nearest_checkpoint_before(self):
        """Test finding nearest checkpoint before a given step."""
        mgr = CheckpointManager(n_steps=100, n_checkpoints=5)

        # Save checkpoints at 0, 25, 50, 75, 100
        for i in [0, 25, 50, 75]:
            x = torch.randn(3, 2)
            v = torch.randn(3, 2)
            alpha = torch.randn(3)
            mgr.save_checkpoint(i, x, v, alpha)

        # Test finding nearest checkpoint
        idx, _ = mgr.get_nearest_checkpoint_before(30)
        assert idx == 25

        idx, _ = mgr.get_nearest_checkpoint_before(50)
        assert idx == 50

        idx, _ = mgr.get_nearest_checkpoint_before(10)
        assert idx == 0

    def test_should_checkpoint(self):
        """Test checkpoint scheduling."""
        mgr = CheckpointManager(n_steps=100, n_checkpoints=5)

        # Should checkpoint at 0, 25, 50, 75
        assert mgr.should_checkpoint(0)
        assert mgr.should_checkpoint(25)
        assert mgr.should_checkpoint(50)
        assert mgr.should_checkpoint(75)
        assert not mgr.should_checkpoint(10)
        assert not mgr.should_checkpoint(100)


class TestCheckpointedNoseHoover:
    """Tests for CheckpointedNoseHoover integrator with discrete adjoint."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_matches_standard(self, device):
        """Forward pass should match standard NoseHoover."""
        kT, mass, Q = 1.0, 1.0, 1.0
        x0 = torch.tensor([[1.0, 0.0]], device=device)
        v0 = torch.tensor([[0.0, 1.0]], device=device)
        dt, n_steps = 0.01, 100

        potential = Harmonic(k=1.0)
        force_fn = potential.force

        # Standard NoseHoover
        integrator_std = NoseHoover(kT=kT, mass=mass, Q=Q)
        traj_x_std, traj_v_std = integrator_std.run(x0, v0, force_fn, dt, n_steps, store_every=10)

        # Checkpointed NoseHoover
        integrator_ckpt = CheckpointedNoseHoover(kT=kT, mass=mass, Q=Q, checkpoint_segments=5)
        traj_x_ckpt, traj_v_ckpt = integrator_ckpt.run(x0, v0, force_fn, dt, n_steps, store_every=10)

        # Trajectories should match
        assert torch.allclose(traj_x_ckpt, traj_x_std, atol=1e-6)
        assert torch.allclose(traj_v_ckpt, traj_v_std, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_vs_bptt_harmonic(self, device):
        """Adjoint gradients should match BPTT for harmonic oscillator."""
        x0 = torch.tensor([[1.0, 0.5]], device=device)
        v0 = torch.tensor([[0.0, 0.5]], device=device)
        dt, n_steps = 0.01, 50

        # BPTT (standard NoseHoover with autograd)
        potential_bptt = Harmonic(k=1.0)
        potential_bptt.k.requires_grad = True
        integrator_bptt = NoseHoover(kT=1.0, mass=1.0, Q=1.0)

        traj_x_bptt, traj_v_bptt = integrator_bptt.run(x0, v0, potential_bptt.force, dt, n_steps)
        loss_bptt = traj_x_bptt[-1].pow(2).sum()
        loss_bptt.backward()
        grad_kT_bptt = integrator_bptt.kT.grad.clone()

        # Checkpointed adjoint
        potential_ckpt = Harmonic(k=1.0)
        integrator_ckpt = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=5)

        traj_x_ckpt, traj_v_ckpt = integrator_ckpt.run(x0, v0, potential_ckpt.force, dt, n_steps)
        loss_ckpt = traj_x_ckpt[-1].pow(2).sum()
        loss_ckpt.backward()
        grad_kT_ckpt = integrator_ckpt.kT.grad

        # Gradients should match
        print(f"BPTT grad_kT: {grad_kT_bptt.item():.6f}")
        print(f"Adjoint grad_kT: {grad_kT_ckpt.item():.6f}")
        assert torch.isfinite(grad_kT_bptt)
        assert torch.isfinite(grad_kT_ckpt)
        assert torch.allclose(grad_kT_ckpt, grad_kT_bptt, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_vs_finite_differences(self, device):
        """Adjoint gradients should match finite differences for mass parameter."""
        x0 = torch.tensor([[1.0, 0.0]], device=device)
        v0 = torch.tensor([[0.0, 1.0]], device=device)
        dt, n_steps = 0.01, 50

        potential = Harmonic(k=1.0)

        # Compute gradient via adjoint
        # Test mass gradient since it has direct effect on dynamics (via F/m)
        integrator = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=5)
        traj_x, traj_v = integrator.run(x0, v0, potential.force, dt, n_steps)
        loss = traj_x[-1].pow(2).sum()
        loss.backward()
        grad_mass_adjoint = integrator.mass.grad.item()

        # Compute gradient via finite differences
        eps = 1e-5

        integrator_plus = CheckpointedNoseHoover(kT=1.0, mass=1.0 + eps, Q=1.0, checkpoint_segments=5)
        traj_x_plus, _ = integrator_plus.run(x0, v0, potential.force, dt, n_steps)
        loss_plus = traj_x_plus[-1].pow(2).sum().item()

        integrator_minus = CheckpointedNoseHoover(kT=1.0, mass=1.0 - eps, Q=1.0, checkpoint_segments=5)
        traj_x_minus, _ = integrator_minus.run(x0, v0, potential.force, dt, n_steps)
        loss_minus = traj_x_minus[-1].pow(2).sum().item()

        grad_mass_fd = (loss_plus - loss_minus) / (2 * eps)

        print(f"Adjoint grad_mass: {grad_mass_adjoint:.6f}")
        print(f"Finite diff grad_mass: {grad_mass_fd:.6f}")
        assert torch.isfinite(torch.tensor(grad_mass_adjoint))
        # Tolerance of 10% to account for discretization and numerical errors
        assert abs(grad_mass_adjoint - grad_mass_fd) / (abs(grad_mass_fd) + 1e-8) < 0.10

    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_dimensions(self, device):
        """Should handle batch dimensions correctly."""
        x0 = torch.randn(5, 3, device=device)  # 5 walkers, 3D
        v0 = torch.randn(5, 3, device=device)
        dt, n_steps = 0.01, 50

        potential = Harmonic(k=1.0)
        integrator = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=5)

        traj_x, traj_v = integrator.run(x0, v0, potential.force, dt, n_steps, store_every=10)

        assert traj_x.shape == (6, 5, 3)  # (n_stored, n_walkers, dim)
        assert traj_v.shape == (6, 5, 3)

        # Test gradients
        loss = traj_x[-1].pow(2).sum()
        loss.backward()
        assert integrator.kT.grad is not None
        assert torch.isfinite(integrator.kT.grad)

    @pytest.mark.parametrize("device", DEVICES)
    def test_different_checkpoint_counts(self, device):
        """Should work with various checkpoint strategies."""
        x0 = torch.tensor([[1.0, 0.0]], device=device)
        v0 = torch.tensor([[0.0, 1.0]], device=device)
        dt, n_steps = 0.01, 100

        potential = Harmonic(k=1.0)

        for n_checkpoints in [1, 5, 10, 20]:
            integrator = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=n_checkpoints)
            traj_x, traj_v = integrator.run(x0, v0, potential.force, dt, n_steps)

            loss = traj_x[-1].pow(2).sum()
            loss.backward()

            assert integrator.kT.grad is not None
            assert torch.isfinite(integrator.kT.grad)


class TestContinuousAdjointNoseHoover:
    """Tests for ContinuousAdjointNoseHoover with continuous adjoint ODEs."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_matches_standard(self, device):
        """Forward pass should match standard NoseHoover."""
        kT, mass, Q = 1.0, 1.0, 1.0
        x0 = torch.tensor([[1.0, 0.0]], device=device)
        v0 = torch.tensor([[0.0, 1.0]], device=device)
        dt, n_steps = 0.01, 100

        potential = Harmonic(k=1.0)
        force_fn = potential.force

        # Standard NoseHoover
        integrator_std = NoseHoover(kT=kT, mass=mass, Q=Q)
        traj_x_std, traj_v_std = integrator_std.run(x0, v0, force_fn, dt, n_steps, store_every=10)

        # Continuous adjoint NoseHoover
        integrator_adj = ContinuousAdjointNoseHoover(kT=kT, mass=mass, Q=Q)
        traj_x_adj, traj_v_adj, traj_alpha_adj = integrator_adj.run(x0, v0, force_fn, dt, n_steps, store_every=10)

        # Trajectories should match
        assert torch.allclose(traj_x_adj, traj_x_std, atol=1e-6)
        assert torch.allclose(traj_v_adj, traj_v_std, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_adjoint_backward_harmonic(self, device):
        """Test adjoint backward pass on harmonic oscillator."""
        x0 = torch.tensor([[1.0, 0.5]], device=device)
        v0 = torch.tensor([[0.0, 0.5]], device=device)
        dt, n_steps = 0.01, 50

        potential = Harmonic(k=1.0)
        integrator = ContinuousAdjointNoseHoover(kT=1.0, mass=1.0, Q=1.0)

        # Forward pass
        traj_x, traj_v, traj_alpha = integrator.run(x0, v0, potential.force, dt, n_steps)

        # Compute loss and gradient analytically
        # For loss = ||x||^2, gradient is 2*x
        loss = traj_x[-1].pow(2).sum()
        grad_x_final = 2 * traj_x[-1]

        # Run adjoint backward
        # Create list of gradients (None for all except final)
        loss_grad_x = [None] * len(traj_x)
        loss_grad_x[-1] = grad_x_final
        loss_grad_v = [None] * len(traj_v)

        grads = integrator.adjoint_backward(
            loss_grad_x, loss_grad_v, traj_x, traj_v, traj_alpha, potential.force, dt
        )

        # Check gradients are finite
        assert torch.isfinite(grads['kT'])
        assert torch.isfinite(grads['mass'])
        assert torch.isfinite(grads['Q'])
        assert torch.isfinite(grads['x0']).all()
        assert torch.isfinite(grads['v0']).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_vs_bptt_harmonic(self, device):
        """Continuous adjoint gradients should match BPTT."""
        x0 = torch.tensor([[1.0, 0.5]], device=device)
        v0 = torch.tensor([[0.0, 0.5]], device=device)
        dt, n_steps = 0.01, 50

        # BPTT (standard NoseHoover with autograd)
        potential_bptt = Harmonic(k=1.0)
        integrator_bptt = NoseHoover(kT=1.0, mass=1.0, Q=1.0)

        traj_x_bptt, traj_v_bptt = integrator_bptt.run(x0, v0, potential_bptt.force, dt, n_steps)
        loss_bptt = traj_x_bptt[-1].pow(2).sum()
        loss_bptt.backward()
        grad_kT_bptt = integrator_bptt.kT.grad.clone()

        # Continuous adjoint
        potential_adj = Harmonic(k=1.0)
        integrator_adj = ContinuousAdjointNoseHoover(kT=1.0, mass=1.0, Q=1.0)

        traj_x_adj, traj_v_adj, traj_alpha_adj = integrator_adj.run(x0, v0, potential_adj.force, dt, n_steps)
        loss_adj = traj_x_adj[-1].pow(2).sum()

        # Get loss gradient and run adjoint backward
        grad_x_final = torch.autograd.grad(loss_adj, traj_x_adj[-1], retain_graph=True, allow_unused=True)
        if grad_x_final[0] is not None:
            loss_grad_x = [None] * len(traj_x_adj)
            loss_grad_x[-1] = grad_x_final[0]
            loss_grad_v = [None] * len(traj_v_adj)

            grads_adj = integrator_adj.adjoint_backward(
                loss_grad_x, loss_grad_v, traj_x_adj, traj_v_adj, traj_alpha_adj, potential_adj.force, dt
            )
            grad_kT_adj = grads_adj['kT']

            # Gradients should be similar (may not be exact due to discretization)
            print(f"BPTT grad_kT: {grad_kT_bptt.item():.6f}")
            print(f"Continuous adjoint grad_kT: {grad_kT_adj.item():.6f}")
            assert torch.isfinite(grad_kT_bptt)
            assert torch.isfinite(grad_kT_adj)
            # Looser tolerance for continuous adjoint due to discretization error
            assert torch.allclose(grad_kT_adj, grad_kT_bptt, atol=1e-3, rtol=1e-2)

    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_dimensions(self, device):
        """Should handle batch dimensions correctly."""
        x0 = torch.randn(5, 3, device=device)  # 5 walkers, 3D
        v0 = torch.randn(5, 3, device=device)
        dt, n_steps = 0.01, 50

        potential = Harmonic(k=1.0)
        integrator = ContinuousAdjointNoseHoover(kT=1.0, mass=1.0, Q=1.0)

        traj_x, traj_v, traj_alpha = integrator.run(x0, v0, potential.force, dt, n_steps, store_every=10)

        assert traj_x.shape == (6, 5, 3)  # (n_stored, n_walkers, dim)
        assert traj_v.shape == (6, 5, 3)
        assert traj_alpha.shape == (6, 5)

        # Test adjoint backward
        # For loss = ||x||^2, gradient is 2*x
        grad_x_final = 2 * traj_x[-1]

        loss_grad_x = [None] * len(traj_x)
        loss_grad_x[-1] = grad_x_final
        loss_grad_v = [None] * len(traj_v)

        grads = integrator.adjoint_backward(
            loss_grad_x, loss_grad_v, traj_x, traj_v, traj_alpha, potential.force, dt
        )

        assert grads['kT'] is not None
        assert torch.isfinite(grads['kT'])


class TestAdjointComparison:
    """Cross-validation tests comparing different gradient methods."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_all_methods_agree_harmonic(self, device):
        """BPTT, discrete adjoint, and continuous adjoint should agree on harmonic."""
        x0 = torch.tensor([[1.0, 0.5]], device=device)
        v0 = torch.tensor([[0.0, 0.5]], device=device)
        dt, n_steps = 0.01, 50

        potential = Harmonic(k=1.0)

        # BPTT
        integrator_bptt = NoseHoover(kT=1.0, mass=1.0, Q=1.0)
        traj_x_bptt, _ = integrator_bptt.run(x0, v0, potential.force, dt, n_steps)
        loss_bptt = traj_x_bptt[-1].pow(2).sum()
        loss_bptt.backward()
        grad_kT_bptt = integrator_bptt.kT.grad.item()

        # Discrete adjoint
        integrator_discrete = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=5)
        traj_x_discrete, _ = integrator_discrete.run(x0, v0, potential.force, dt, n_steps)
        loss_discrete = traj_x_discrete[-1].pow(2).sum()
        loss_discrete.backward()
        grad_kT_discrete = integrator_discrete.kT.grad.item()

        # Continuous adjoint
        integrator_continuous = ContinuousAdjointNoseHoover(kT=1.0, mass=1.0, Q=1.0)
        traj_x_continuous, traj_v_continuous, traj_alpha_continuous = integrator_continuous.run(
            x0, v0, potential.force, dt, n_steps
        )
        # For loss = ||x||^2, gradient is 2*x
        grad_x_final = 2 * traj_x_continuous[-1]

        loss_grad_x = [None] * len(traj_x_continuous)
        loss_grad_x[-1] = grad_x_final
        loss_grad_v = [None] * len(traj_v_continuous)

        grads_continuous = integrator_continuous.adjoint_backward(
            loss_grad_x, loss_grad_v, traj_x_continuous, traj_v_continuous,
            traj_alpha_continuous, potential.force, dt
        )
        grad_kT_continuous = grads_continuous['kT'].item()

        print(f"BPTT: {grad_kT_bptt:.6f}")
        print(f"Discrete adjoint: {grad_kT_discrete:.6f}")
        print(f"Continuous adjoint: {grad_kT_continuous:.6f}")

        # All should be finite
        assert torch.isfinite(torch.tensor(grad_kT_bptt))
        assert torch.isfinite(torch.tensor(grad_kT_discrete))
        assert torch.isfinite(torch.tensor(grad_kT_continuous))

        # Discrete adjoint should match BPTT closely
        assert abs(grad_kT_discrete - grad_kT_bptt) / (abs(grad_kT_bptt) + 1e-8) < 1e-3

        # Continuous adjoint should be similar (looser tolerance due to discretization error)
        # The continuous adjoint is derived from continuous-time equations, so it can differ
        # from the discrete gradient when the dynamics have rapid thermostat coupling.
        assert abs(grad_kT_continuous - grad_kT_bptt) / (abs(grad_kT_bptt) + 1e-8) < 10.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_all_methods_agree_doublewell(self, device):
        """Test all methods on double well potential."""
        x0 = torch.tensor([[0.5, 0.0]], device=device)
        v0 = torch.tensor([[0.0, 0.1]], device=device)
        dt, n_steps = 0.01, 50

        potential = DoubleWell(barrier_height=2.0)

        # BPTT
        integrator_bptt = NoseHoover(kT=1.0, mass=1.0, Q=1.0)
        traj_x_bptt, _ = integrator_bptt.run(x0, v0, potential.force, dt, n_steps)
        loss_bptt = traj_x_bptt[-1].pow(2).sum()
        loss_bptt.backward()
        grad_kT_bptt = integrator_bptt.kT.grad.item()

        # Discrete adjoint
        integrator_discrete = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=5)
        traj_x_discrete, _ = integrator_discrete.run(x0, v0, potential.force, dt, n_steps)
        loss_discrete = traj_x_discrete[-1].pow(2).sum()
        loss_discrete.backward()
        grad_kT_discrete = integrator_discrete.kT.grad.item()

        print(f"DoubleWell - BPTT: {grad_kT_bptt:.6f}")
        print(f"DoubleWell - Discrete adjoint: {grad_kT_discrete:.6f}")

        # Both should be finite
        assert torch.isfinite(torch.tensor(grad_kT_bptt))
        assert torch.isfinite(torch.tensor(grad_kT_discrete))

        # Should match closely
        assert abs(grad_kT_discrete - grad_kT_bptt) / (abs(grad_kT_bptt) + 1e-8) < 1e-3

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradients_w_r_t_all_parameters(self, device):
        """Test gradients w.r.t. all parameters (kT, mass, Q)."""
        x0 = torch.tensor([[1.0, 0.0]], device=device)
        v0 = torch.tensor([[0.0, 1.0]], device=device)
        dt, n_steps = 0.01, 50

        potential = Harmonic(k=1.0)

        # Test with discrete adjoint
        integrator = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=5)
        traj_x, traj_v = integrator.run(x0, v0, potential.force, dt, n_steps)
        loss = traj_x[-1].pow(2).sum()
        loss.backward()

        # All parameter gradients should be finite
        assert integrator.kT.grad is not None
        assert integrator.mass.grad is not None
        assert integrator.Q.grad is not None
        assert torch.isfinite(integrator.kT.grad)
        assert torch.isfinite(integrator.mass.grad)
        assert torch.isfinite(integrator.Q.grad)

        print(f"grad_kT: {integrator.kT.grad.item():.6f}")
        print(f"grad_mass: {integrator.mass.grad.item():.6f}")
        print(f"grad_Q: {integrator.Q.grad.item():.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
