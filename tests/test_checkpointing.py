
import pytest
import torch
import torch.nn as nn
from uni_diffsim.gradient_estimators import StochasticAdjoint, CheckpointedNoseHoover
from uni_diffsim.potentials import Harmonic
from uni_diffsim.integrators import OverdampedLangevin, NoseHoover
from uni_diffsim.device import available_devices

DEVICES = available_devices()

class TestStochasticAdjoint:
    @pytest.mark.parametrize("device", DEVICES)
    def test_gradients_match_bptt(self, device):
        """Compare StochasticAdjoint gradients with standard BPTT."""
        torch.manual_seed(42)

        # Setup common parameters
        gamma = 1.0
        kT = 1.0
        dt = 0.01
        n_steps = 50
        segment_length = 10

        # 1. Standard BPTT (using retain_graph or just backprop through operations)
        potential1 = Harmonic(k=1.0).to(device)
        integrator1 = OverdampedLangevin(gamma=gamma, kT=kT).to(device)

        # Use same initial state
        x0_base = torch.randn(10, 1, device=device)
        x0_1 = x0_base.clone().requires_grad_(True)

        # Fix noise by manual seeding inside the run?
        # StochasticAdjoint and OverdampedLangevin use torch.randn_like, which uses global state.
        # To match EXACTLY, we need to control the RNG state sequence.
        # This is tricky because StochasticAdjoint runs segments and potentially re-runs them.
        # But for the FIRST pass, they should consume RNG in same order if implemented identically.
        # Wait, StochasticAdjoint uses 'checkpoint', which runs forward in no_grad mode,
        # then backward re-runs forward.
        # Standard BPTT runs forward once.
        # The noise drawn in the first forward pass must match.

        # Let's seed before running
        torch.manual_seed(123)
        traj1 = integrator1.run(x0_1, potential1.force, dt, n_steps)
        loss1 = traj1[-1].sum()
        loss1.backward()
        grad_k1 = potential1.k.grad.clone()
        grad_x0_1 = x0_1.grad.clone()

        # 2. StochasticAdjoint
        potential2 = Harmonic(k=1.0).to(device)
        # We need to ensure integrator parameters match (they do by default init)
        stoch_adj = StochasticAdjoint(gamma=gamma, kT=kT, segment_length=segment_length).to(device)
        # Share potential instance? No, we want to check its gradients.

        x0_2 = x0_base.clone().requires_grad_(True)

        # Seed must be same.
        torch.manual_seed(123)
        # Run
        # Note: StochasticAdjoint.run uses checkpointing.
        # During the forward pass, it draws noise.
        # Since logic is: for segment in segments: x = checkpoint(run_segment, ...)
        # And run_segment calls integrator.step which draws noise.
        # The sequence of RNG calls should be identical to standard run loop.
        traj2 = stoch_adj.run(x0_2, potential2.force, dt, n_steps)

        # Check trajectories match forward
        assert torch.allclose(traj1[-1], traj2[-1], atol=1e-5), "Forward trajectories should match"

        loss2 = traj2[-1].sum()
        loss2.backward()
        grad_k2 = potential2.k.grad.clone()
        grad_x0_2 = x0_2.grad.clone()

        # Gradients should match exactly (or very close due to float arithmetic order)
        # Note: checkpointing might introduce slight numerical differences due to re-computation order?
        # Usually exact for determinstic, but for stochastic, we rely on preserve_rng_state.
        assert torch.allclose(grad_k1, grad_k2, atol=1e-5), f"Potential gradients mismatch: {grad_k1} vs {grad_k2}"
        assert torch.allclose(grad_x0_1, grad_x0_2, atol=1e-5), "Input gradients mismatch"

    @pytest.mark.parametrize("device", DEVICES)
    def test_final_only_shape(self, device):
        stoch_adj = StochasticAdjoint(segment_length=5).to(device)
        potential = Harmonic(k=1.0).to(device)
        x0 = torch.randn(5, 1, device=device)

        traj = stoch_adj.run(x0, potential.force, dt=0.01, n_steps=20, final_only=True)
        assert traj.shape == (1, 5, 1) # (1, batch, dim)


class TestCheckpointedNoseHoover:
    @pytest.mark.parametrize("device", DEVICES)
    def test_potential_gradients_exist(self, device):
        """Verify that gradients flow to potential parameters (fixing the bug)."""
        torch.manual_seed(42)
        potential = Harmonic(k=1.0).to(device)
        integrator = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, segment_length=5).to(device)

        x0 = torch.randn(10, 1, device=device, requires_grad=True)
        v0 = torch.randn(10, 1, device=device)

        traj_x, traj_v = integrator.run(x0, v0, potential.force, dt=0.01, n_steps=20)

        loss = traj_x[-1].sum()
        loss.backward()

        assert potential.k.grad is not None, "Potential parameter gradient should not be None"
        assert x0.grad is not None
        assert integrator.integrator.kT.grad is not None # Integrator params also

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradients_match_standard(self, device):
        """Compare with standard NoseHoover BPTT."""
        torch.manual_seed(42)
        potential1 = Harmonic(k=2.0).to(device)
        integrator1 = NoseHoover(kT=1.0, mass=1.0, Q=1.0).to(device)
        x0 = torch.randn(5, 1, device=device)
        x0_1 = x0.clone().requires_grad_(True)
        v0_1 = torch.zeros_like(x0_1)

        traj1_x, _ = integrator1.run(x0_1, v0_1, potential1.force, dt=0.01, n_steps=20)
        loss1 = traj1_x[-1].sum()
        loss1.backward()
        grad1 = potential1.k.grad.clone()

        # Checkpointed
        torch.manual_seed(42) # Reset for consistency if needed (though NH is deterministic)
        potential2 = Harmonic(k=2.0).to(device)
        integrator2 = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, segment_length=5).to(device)
        x0_2 = x0.clone().requires_grad_(True)
        v0_2 = torch.zeros_like(x0_2)

        traj2_x, _ = integrator2.run(x0_2, v0_2, potential2.force, dt=0.01, n_steps=20)
        loss2 = traj2_x[-1].sum()
        loss2.backward()
        grad2 = potential2.k.grad.clone()

        assert torch.allclose(grad1, grad2, atol=1e-5)
        # Checkpointed trajectory is sparse (only boundaries)
        # Check matching at the final step
        assert torch.allclose(traj1_x[-1], traj2_x[-1], atol=1e-5)
        # Check matching at start
        assert torch.allclose(traj1_x[0], traj2_x[0], atol=1e-5)
