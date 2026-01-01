
import pytest
import torch
import torch.nn as nn
from uni_diffsim.gradient_estimators import ContinuousAdjointNoseHoover
from uni_diffsim.potentials import Harmonic
from uni_diffsim.device import available_devices

DEVICES = available_devices()

class TestContinuousAdjointO1:
    @pytest.mark.parametrize("device", DEVICES)
    def test_o1_vs_ot_gradients(self, device):
        """Verify O(1) memory method matches O(T) memory method."""
        torch.manual_seed(42)

        # Setup
        kT = 1.0
        mass = 1.0
        Q = 1.0
        dt = 0.01
        n_steps = 20

        potential = Harmonic(k=2.0).to(device)
        # We need two integrators to avoid parameter sharing issues in testing logic
        # But we want to test the SAME integrator instance usually.
        # Let's use one instance but reset gradients.

        integrator = ContinuousAdjointNoseHoover(kT, mass, Q).to(device)

        # Initial state
        x0 = torch.randn(5, 1, device=device)
        v0 = torch.randn(5, 1, device=device)

        # 1. Run forward and store trajectory (O(T))
        traj_x, traj_v, traj_alpha = integrator.run(x0, v0, potential.force, dt, n_steps)

        # Fake loss gradient at final step
        loss_grad_x = [None] * n_steps + [torch.ones_like(traj_x[-1])]
        loss_grad_v = [None] * (n_steps + 1)

        # 2. Compute gradients using O(T) mode
        grads_ot = integrator.adjoint_backward(
            loss_grad_x, loss_grad_v,
            traj_x, traj_v, traj_alpha,
            potential.force, dt
        )

        # 3. Compute gradients using O(1) mode (reconstruction)
        final_state = (traj_x[-1], traj_v[-1], traj_alpha[-1])

        grads_o1 = integrator.adjoint_backward(
            loss_grad_x, loss_grad_v,
            None, None, None, # traj is None
            potential.force, dt,
            final_state=final_state,
            n_steps=n_steps
        )

        # Compare
        for name in grads_ot:
            if name in ['x0', 'v0', 'alpha0']: # Initial state gradients
                # Initial state should match
                # Note: reconstruction numerical error might be small but present
                assert torch.allclose(grads_ot[name], grads_o1[name], atol=1e-5), f"Gradient {name} mismatch"
            else: # Parameter gradients
                assert torch.allclose(grads_ot[name], grads_o1[name], atol=1e-5), f"Gradient {name} mismatch"

    @pytest.mark.parametrize("device", DEVICES)
    def test_reverse_step_exactness(self, device):
        """Verify reverse_step exactly inverts step."""
        torch.manual_seed(123)
        integrator = ContinuousAdjointNoseHoover(kT=1.0, mass=1.0, Q=1.0).to(device)
        potential = Harmonic(k=1.0).to(device)

        x = torch.randn(10, 1, device=device)
        v = torch.randn(10, 1, device=device)
        alpha = torch.zeros(10, 1, device=device)
        dt = 0.1

        # Forward
        x_new, v_new, alpha_new = integrator.step(x, v, alpha, potential.force, dt)

        # Reverse
        x_rec, v_rec, alpha_rec = integrator.reverse_step(x_new, v_new, alpha_new, potential.force, dt)

        assert torch.allclose(x, x_rec, atol=1e-5)
        assert torch.allclose(v, v_rec, atol=1e-5)
        assert torch.allclose(alpha, alpha_rec, atol=1e-5)
