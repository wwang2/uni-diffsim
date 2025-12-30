
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from uni_diffsim.potentials import (
    DoubleWell, DoubleWell2D, MullerBrown, LennardJones, Harmonic
)

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

assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(assets_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
fig.patch.set_facecolor('#FAFBFC')

# Common colormap and style
cmap = 'BuPu'
particle_color = '#5E81AC'  # Steel blue from Nord palette

# 1. Double Well 2D
ax = axes[0, 0]
dw = DoubleWell2D()
x = torch.linspace(-2.5, 2.5, 100)
y = torch.linspace(-2.0, 2.0, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')
xy = torch.stack([X, Y], dim=-1)
U = dw.energy(xy)

# Levels for DoubleWell2D
levels = np.linspace(0, 5, 21)
U_clipped = torch.clamp(U, max=5.0)

cs = ax.contourf(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, cmap=cmap, extend='max')
ax.contour(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, colors='k', linewidths=0.3, alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Double Well 2D', fontweight='bold')
plt.colorbar(cs, ax=ax, label='U')
ax.set_axisbelow(True)

# 2. Müller-Brown contour
ax = axes[0, 1]
mb = MullerBrown()
x = torch.linspace(-1.5, 1.2, 100)
y = torch.linspace(-0.5, 2.0, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')
xy = torch.stack([X, Y], dim=-1)
U = mb.energy(xy)

levels = np.linspace(-150, 100, 26)
U_clipped = torch.clamp(U, max=100.0)

cs = ax.contourf(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, cmap=cmap, extend='max')
ax.contour(X.numpy(), Y.numpy(), U_clipped.detach().numpy(), levels=levels, colors='k', linewidths=0.3, alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Müller-Brown', fontweight='bold')
plt.colorbar(cs, ax=ax, label='U')
ax.set_axisbelow(True)

# 3. LJ-7 cluster (no PBC)
ax = axes[1, 0]
lj = LennardJones()
angles = torch.linspace(0, 2*np.pi, 7)[:-1]
r = 1.12
positions = torch.zeros(7, 2)
positions[1:, 0] = r * torch.cos(angles)
positions[1:, 1] = r * torch.sin(angles)

ax.scatter(positions[:, 0].numpy(), positions[:, 1].numpy(),
           s=500, c=particle_color, edgecolor='k', lw=1.5, zorder=10)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_title(f'LJ-7 cluster (U={lj.energy(positions).item():.2f})', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_axisbelow(True)

# 4. LJ cluster with periodic boundary conditions
ax = axes[1, 1]
L = 4.0  # box size
lj_pbc = LennardJones(box_size=L)

# Create a 3x3 grid of particles in the box
n_side = 3
spacing = L / n_side
pos_pbc = torch.zeros(n_side * n_side, 2)
idx = 0
for i in range(n_side):
    for j in range(n_side):
        pos_pbc[idx, 0] = (i + 0.5) * spacing
        pos_pbc[idx, 1] = (j + 0.5) * spacing
        idx += 1

# Draw the periodic box
box = plt.Rectangle((0, 0), L, L, fill=False, edgecolor='#333', lw=2, ls='--')
ax.add_patch(box)

# Draw ghost images (periodic copies) in faded color
for dx in [-L, 0, L]:
    for dy in [-L, 0, L]:
        if dx == 0 and dy == 0:
            continue  # skip the main box
        ghost_pos = pos_pbc + torch.tensor([dx, dy])
        ax.scatter(ghost_pos[:, 0].numpy(), ghost_pos[:, 1].numpy(),
                  s=200, c=particle_color, alpha=0.2, edgecolor='none', zorder=5)

# Draw main particles
ax.scatter(pos_pbc[:, 0].numpy(), pos_pbc[:, 1].numpy(),
           s=400, c=particle_color, edgecolor='k', lw=1.5, zorder=10)

ax.set_xlim(-L*0.5, L*1.5)
ax.set_ylim(-L*0.5, L*1.5)
ax.set_aspect('equal')
ax.set_title(f'LJ-9 with PBC (L={L}, U={lj_pbc.energy(pos_pbc).item():.2f})', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_axisbelow(True)

plt.savefig(os.path.join(assets_dir, "potentials.png"), dpi=150,
            bbox_inches='tight', facecolor='#FAFBFC')
print(f"Saved potentials plot to assets/potentials.png")
