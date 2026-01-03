
import pytest
import torch
from uni_diffsim.potentials import LennardJones, LennardJonesVerlet
from uni_diffsim.device import available_devices

DEVICES = available_devices()

class TestLennardJonesVerletBatched:
    """Tests for batched support in LennardJonesVerlet."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_compare_dense_batched(self, device):
        """Compare batched Verlet implementation with dense O(N^2) implementation."""
        # Setup parameters
        eps = 1.0
        sigma = 1.0
        # Use large cutoff to ensure all pairs are included for comparison
        cutoff = 10.0
        skin = 0.5

        lj_dense = LennardJones(eps=eps, sigma=sigma).to(device)
        lj_verlet = LennardJonesVerlet(eps=eps, sigma=sigma, cutoff=cutoff, skin=skin).to(device)

        # Create batched input: (B=3, N=5, D=2)
        batch_size = 3
        n_particles = 5
        dim = 2
        x = torch.randn(batch_size, n_particles, dim, device=device) * 1.5

        # Ensure gradients work
        x.requires_grad_(True)

        # Compute dense energy
        u_dense = lj_dense.energy(x)

        # Compute verlet energy
        # First update neighbor list
        lj_verlet.update_neighbor_list(x)
        u_verlet = lj_verlet.energy(x)

        assert u_dense.shape == u_verlet.shape == (batch_size,)
        assert torch.allclose(u_dense, u_verlet, atol=1e-5), "Energies do not match"

        # Check gradients
        grad_dense = torch.autograd.grad(u_dense.sum(), x, retain_graph=True)[0]
        grad_verlet = torch.autograd.grad(u_verlet.sum(), x, retain_graph=True)[0]

        assert torch.allclose(grad_dense, grad_verlet, atol=1e-5), "Gradients do not match"

    @pytest.mark.parametrize("device", DEVICES)
    def test_compare_dense_multidim_batch(self, device):
        """Compare batched Verlet implementation with multidimensional batch."""
        # Setup parameters
        eps = 1.0
        sigma = 1.0
        cutoff = 10.0
        skin = 0.5

        lj_dense = LennardJones(eps=eps, sigma=sigma).to(device)
        lj_verlet = LennardJonesVerlet(eps=eps, sigma=sigma, cutoff=cutoff, skin=skin).to(device)

        # Create multidim batched input: (B1=2, B2=2, N=4, D=2)
        x = torch.randn(2, 2, 4, 2, device=device) * 1.5

        # Update neighbor list
        lj_verlet.update_neighbor_list(x)

        u_dense = lj_dense.energy(x)
        u_verlet = lj_verlet.energy(x)

        assert u_dense.shape == (2, 2)
        assert u_verlet.shape == (2, 2)
        assert torch.allclose(u_dense, u_verlet, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_cutoff_behavior(self, device):
        """Test that Verlet list respects cutoff (pairs beyond cutoff should be excluded if we strictly implemented it,
        but current implementation calculates all in neighbor list).

        Here we just test consistency with different batch elements having different configurations.
        """
        lj_verlet = LennardJonesVerlet(cutoff=2.0, skin=0.5).to(device)

        # Batch of 2 systems
        # System 1: Two particles close (r=1.0)
        # System 2: Two particles far (r=5.0) -> should have 0 energy if neighbors not found

        x = torch.zeros(2, 2, 2, device=device)
        x[0, 1, 0] = 1.122 # r~r_eq, energy should be -1.0 (approx)
        x[1, 1, 0] = 5.0 # r=5.0 > 2.5

        lj_verlet.update_neighbor_list(x)
        u = lj_verlet.energy(x)

        # System 1 should have energy
        assert u[0] != 0
        # System 2 should have 0 energy (no neighbors found)
        # Note: if neighbors ARE found (e.g. huge skin), energy would be small but non-zero.
        # With cutoff 2.0+0.5=2.5, r=5.0 is definitely out.
        assert u[1] == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_displacement_check_batched(self, device):
        """Test that check_neighbor_list works for batched inputs."""
        lj_verlet = LennardJonesVerlet(skin=1.0).to(device)

        # Initial config
        x = torch.randn(2, 4, 2, device=device)
        lj_verlet.update_neighbor_list(x)

        # Small displacement -> should return False
        x_small = x + torch.randn_like(x) * 0.01
        assert not lj_verlet.check_neighbor_list(x_small)

        # Large displacement in one batch element -> should return True
        x_large = x.clone()
        x_large[1, 0, 0] += 10.0
        assert lj_verlet.check_neighbor_list(x_large)
