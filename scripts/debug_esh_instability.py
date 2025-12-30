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

# Plotting style (Nord-inspired, editorial)
plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["JetBrains Mono", "DejaVu Sans Mono", "Menlo", "Monaco"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 8.0,
    "axes.labelpad": 5.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": '0.9',
    "figure.facecolor": "#FAFBFC",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FAFBFC",
    "lines.linewidth": 2.0,
})

# Color palette
COLORS = {
    'ESH': '#B48EAD',          # Lavender
    'Theory': '#4C566A',       # Slate gray
    'Trajectory': '#88C0D0',     # Cyan
    'Error': '#BF616A',          # Red
}

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#FAFBFC')

ax = axes[0]
for i in range(n_chains):
    ax.plot(r[:, i], label=f'chain {i}', color=plt.cm.viridis(i / n_chains), alpha=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('r = log|v|')
ax.set_title('Log velocity magnitude (diverges to -∞)')
ax.legend()

ax = axes[1]
X, Y = torch.meshgrid(torch.linspace(-3, 3, 50), torch.linspace(-3, 3, 50), indexing='ij')
U = dw2d.energy(torch.stack([X, Y], dim=-1))
ax.contour(X.numpy(), Y.numpy(), U.detach().numpy(), levels=20, alpha=0.3, colors='#4C566A')
for i in range(n_chains):
    ax.plot(x[:, i, 0], x[:, i, 1], alpha=0.6, color=plt.cm.viridis(i / n_chains))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trajectories (some escape to high-gradient regions)')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('assets/esh_debug.png', dpi=150, facecolor='#FAFBFC')
print(f"\nSaved: assets/esh_debug.png")

