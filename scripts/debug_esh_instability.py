#!/usr/bin/env python
"""Demonstrate ESH numerical instability.

The ESH integrator diverges when:
  eps × |grad| / d >> 1  AND  u·(-e) ≈ -1

Run: python scripts/debug_esh_instability.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from uni_diffsim.integrators import ESH
from uni_diffsim.potentials import DoubleWell2D

# Setup
dw2d = DoubleWell2D(barrier_height=1.0, k_y=1.0)
grad_fn = lambda x: -dw2d.force(x)

torch.manual_seed(42)
n_chains = 5
x0 = torch.randn(n_chains, 2)
u0 = torch.randn(n_chains, 2)
u0 = u0 / u0.norm(dim=-1, keepdim=True)

# Run ESH
esh = ESH(eps=0.1, max_grad_norm=None)
traj_x, traj_u, traj_r = esh.run(x0, u0, grad_fn, n_steps=2000, dt=0.1, store_every=1)

r = traj_r.detach().numpy()
x = traj_x.detach().numpy()

# Find divergence points
print("ESH Instability Analysis")
print("=" * 50)
for i in range(n_chains):
    diverge_idx = np.where(r[:, i] < -10)[0]
    if len(diverge_idx) > 0:
        step = diverge_idx[0]
        xi = torch.tensor(x[step, i:i+1])
        grad = grad_fn(xi)
        g_norm = grad.norm().item()
        stability = 0.1 * g_norm / 2  # eps × |grad| / d
        print(f"Chain {i}: diverges at step {step}")
        print(f"  x = {x[step, i]}")
        print(f"  |grad| = {g_norm:.1f}")
        print(f"  eps×|grad|/d = {stability:.2f} (should be << 1)")
    else:
        print(f"Chain {i}: stable (r ∈ [{r[:, i].min():.1f}, {r[:, i].max():.1f}])")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for i in range(n_chains):
    ax.plot(r[:, i], label=f'chain {i}')
ax.set_xlabel('Step')
ax.set_ylabel('r = log|v|')
ax.set_title('Log velocity magnitude (diverges to -∞)')
ax.legend()

ax = axes[1]
X, Y = torch.meshgrid(torch.linspace(-3, 3, 50), torch.linspace(-3, 3, 50), indexing='ij')
U = dw2d.energy(torch.stack([X, Y], dim=-1))
ax.contour(X.numpy(), Y.numpy(), U.detach().numpy(), levels=20, alpha=0.3)
for i in range(n_chains):
    ax.plot(x[:, i, 0], x[:, i, 1], alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trajectories (some escape to high-gradient regions)')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('assets/esh_debug.png', dpi=150)
print(f"\nSaved: assets/esh_debug.png")

