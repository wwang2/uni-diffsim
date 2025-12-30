
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from uni_diffsim.potentials import DoubleWell, DoubleWell2D, Harmonic
from uni_diffsim.integrators import (
    OverdampedLangevin, BAOAB, VelocityVerlet, NoseHoover,
    NoseHooverChain, ESH, GLE
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

fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
fig.patch.set_facecolor('#FAFBFC')

# Color palette
COLORS = {
    'Overdamped': '#5E81AC',   # Steel blue
    'BAOAB': '#D08770',        # Warm orange
    'GLE': '#A3BE8C',          # Sage green
    'NH': '#BF616A',           # Muted red
    'ESH': '#B48EAD',          # Lavender
    'Verlet': '#4C566A',       # Slate gray
    'NHC': '#88C0D0',          # Cyan
}

# Setup for 1D double well
dw = DoubleWell()
kT = 0.5
dt = 0.01
n_steps = 50000
n_batch = 50

def force_fn_1d(x):
    return dw.force(x.unsqueeze(-1)).squeeze(-1)

def add_1d_density_inset(ax, samples, potential, kT, color, label=''):
    # Inset for 1D density
    ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])

    # Empirical density (histogram)
    ax_ins.hist(samples, bins=40, density=True, range=(-2.5, 2.5),
               color=color, alpha=0.6, edgecolor='none')

    # Theoretical density
    x_th = torch.linspace(-2.5, 2.5, 200)
    u_th = potential.energy(x_th)
    p_th = torch.exp(-u_th / kT)
    p_th = p_th / (p_th.sum() * (x_th[1] - x_th[0]))
    ax_ins.plot(x_th.numpy(), p_th.detach().numpy(), 'k-', lw=1.5, alpha=0.8)

    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_facecolor('none')
    for spine in ax_ins.spines.values():
        spine.set_visible(False)

# 1. Overdamped Langevin
ax = axes[0, 0]
integrator = OverdampedLangevin(gamma=1.0, kT=kT)
x0 = torch.full((n_batch,), -1.0)
traj_od = integrator.run(x0, force_fn_1d, dt, n_steps, store_every=10)
t = np.arange(traj_od.shape[0]) * dt * 10
for i in range(min(3, n_batch)):
    ax.plot(t, traj_od[:, i].detach().numpy(), alpha=0.75, lw=1.8, color=COLORS['Overdamped'])
ax.axhline(1, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.axhline(-1, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_title('Overdamped Langevin', fontweight='bold')
ax.set_axisbelow(True)
# Add density inset
burn_in = 2000
samples_od = traj_od[burn_in//10:].flatten().detach().numpy()
add_1d_density_inset(ax, samples_od, dw, kT, COLORS['Overdamped'])

# 2. BAOAB
ax = axes[0, 1]
integrator = BAOAB(gamma=1.0, kT=kT, mass=1.0)
traj_baoab, _ = integrator.run(x0, None, force_fn_1d, dt, n_steps, store_every=10)
for i in range(min(3, n_batch)):
    ax.plot(t, traj_baoab[:, i].detach().numpy(), alpha=0.75, lw=1.8, color=COLORS['BAOAB'])
ax.axhline(1, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.axhline(-1, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_title('BAOAB', fontweight='bold')
ax.set_axisbelow(True)
# Add density inset
samples_baoab = traj_baoab[burn_in//10:].flatten().detach().numpy()
add_1d_density_inset(ax, samples_baoab, dw, kT, COLORS['BAOAB'])

# 3. GLE (colored noise)
ax = axes[0, 2]
gle = GLE(kT=kT, mass=1.0, gamma=[0.5, 2.0], c=[0.3, 1.0])
traj_gle, _ = gle.run(x0, None, force_fn_1d, dt, n_steps, store_every=10)
for i in range(min(3, n_batch)):
    ax.plot(t, traj_gle[:, i].detach().numpy(), alpha=0.75, lw=1.8, color=COLORS['GLE'])
ax.axhline(1, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.axhline(-1, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_title('GLE (colored noise)', fontweight='bold')
ax.set_axisbelow(True)
# Add density inset
samples_gle = traj_gle[burn_in//10:].flatten().detach().numpy()
add_1d_density_inset(ax, samples_gle, dw, kT, COLORS['GLE'])

# 4. 2D Double Well Sampling (Row 2)
# Use DoubleWell2D instead of Harmonic
dw2d = DoubleWell2D(barrier_height=1.0, k_y=1.0)
kT_2d = 0.5 # Lower temperature to see hopping

def grad_dw2d(x):
    return -dw2d.force(x)

# Prepare background contours
x_grid = torch.linspace(-2.0, 2.0, 100)
y_grid = torch.linspace(-2.0, 2.0, 100)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
xy_grid = torch.stack([X, Y], dim=-1)
U_grid = dw2d.energy(xy_grid)

# Helper for 2D KDE insets
def add_2d_kde_inset(ax, samples, weights=None, color_map='Blues'):
    ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])

    # Kernel Density Estimation
    try:
        # Subsample for KDE if too large
        if len(samples) > 5000:
            idx = np.random.choice(len(samples), 5000, p=weights if weights is not None else None, replace=False)
            kde_samples = samples[idx]
            kde_weights = weights[idx] if weights is not None else None
        else:
            kde_samples = samples
            kde_weights = weights

        x = kde_samples[:, 0]
        y = kde_samples[:, 1]

        # Create grid for KDE evaluation
        xmin, xmax = -2.0, 2.0
        ymin, ymax = -2.0, 2.0
        xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        kernel = gaussian_kde(values, weights=kde_weights)
        f = np.reshape(kernel(positions).T, xx.shape)

        ax_ins.contourf(xx, yy, f, cmap=color_map, levels=10)
        ax_ins.set_xlim(xmin, xmax)
        ax_ins.set_ylim(ymin, ymax)
    except Exception:
        # Fallback if KDE fails (e.g. singular matrix)
        pass

    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_facecolor('white')
    for spine in ax_ins.spines.values():
        spine.set_visible(True) # Keep box for 2D density

# Run samplers on 2D Double Well
n_steps_2d = 50000

# ESH
torch.manual_seed(42)
esh = ESH(eps=0.1)
x0_esh = torch.randn(20, 2)
u0_esh = torch.randn(20, 2)
u0_esh = u0_esh / u0_esh.norm(dim=-1, keepdim=True)

traj_esh_2d, _, traj_r = esh.run(x0_esh, u0_esh, grad_dw2d, n_steps=n_steps_2d, dt=0.1, store_every=1)

burn_in = 5000
esh_x = traj_esh_2d[burn_in:].detach().numpy()
esh_r = traj_r[burn_in:].detach().numpy()
esh_weights = np.exp(esh_r)
esh_weights = esh_weights / esh_weights.sum()
esh_samples = esh_x.reshape(-1, 2)
esh_w_flat = esh_weights.flatten()

# Nosé-Hoover
nh_2d = NoseHoover(kT=kT_2d, mass=1.0, Q=1.0)
x0_nh = torch.randn(20, 2)
traj_nh_2d, _ = nh_2d.run(x0_nh, None, dw2d.force, dt=0.05, n_steps=n_steps_2d, store_every=1)
nh_samples = traj_nh_2d[5000:].reshape(-1, 2).detach().numpy()

# BAOAB
baoab_2d = BAOAB(gamma=1.0, kT=kT_2d, mass=1.0)
x0_baoab = torch.randn(200, 2) # More chains for baoab to cover space
traj_baoab_2d, _ = baoab_2d.run(x0_baoab, None, dw2d.force, dt=0.05, n_steps=10000, store_every=10)
baoab_samples = traj_baoab_2d[100:].reshape(-1, 2).detach().numpy()

# Plotting 2D
# ESH
ax = axes[1, 0]
ax.contour(X.numpy(), Y.numpy(), U_grid.detach().numpy(), levels=np.linspace(0, 5, 10), colors='k', alpha=0.2)

# Importance resampling for scatter
esh_idx = np.random.choice(len(esh_samples), size=5000, p=esh_w_flat)
ax.scatter(esh_samples[esh_idx, 0], esh_samples[esh_idx, 1], s=5, alpha=0.3,
           c=COLORS['ESH'], edgecolors='none')

ax.set_title('ESH (2D Double Well)', fontweight='bold')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
add_2d_kde_inset(ax, esh_samples, esh_w_flat, color_map='Purples')

# NH
ax = axes[1, 1]
ax.contour(X.numpy(), Y.numpy(), U_grid.detach().numpy(), levels=np.linspace(0, 5, 10), colors='k', alpha=0.2)
ax.scatter(nh_samples[::10, 0], nh_samples[::10, 1], s=5, alpha=0.3,
           c=COLORS['NH'], edgecolors='none')
ax.set_title('Nosé-Hoover (2D Double Well)', fontweight='bold')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
add_2d_kde_inset(ax, nh_samples, None, color_map='Reds')

# BAOAB
ax = axes[1, 2]
ax.contour(X.numpy(), Y.numpy(), U_grid.detach().numpy(), levels=np.linspace(0, 5, 10), colors='k', alpha=0.2)
ax.scatter(baoab_samples[::2, 0], baoab_samples[::2, 1], s=5, alpha=0.3,
           c=COLORS['BAOAB'], edgecolors='none')
ax.set_title('BAOAB (2D Double Well)', fontweight='bold')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
add_2d_kde_inset(ax, baoab_samples, None, color_map='Oranges')

# 7. Benchmark Plot (Forward vs Backward)
ax = axes[2, 0]
# Spanning all columns in the bottom row
ax.remove()
ax = axes[2, 1]
ax.remove()
ax = axes[2, 2]
ax.remove()

# Create a new axis spanning the bottom row
gs = axes[0, 0].get_gridspec()
ax_bench = fig.add_subplot(gs[2, :])

def run_benchmark_for_plot():
    # Setup for benchmark
    dim = 64
    n_particles = 256
    n_steps = 100
    dt = 0.01
    device = torch.device('cpu')

    integrators_list = [
        ("Overdamped", OverdampedLangevin(gamma=1.0, kT=1.0)),
        ("BAOAB", BAOAB(gamma=1.0, kT=1.0, mass=1.0)),
        ("Verlet", VelocityVerlet(mass=1.0)),
        ("NH", NoseHoover(kT=1.0, mass=1.0, Q=1.0)),
        ("NHC", NoseHooverChain(kT=1.0, mass=1.0, Q=1.0, n_chain=2)),
        ("ESH", ESH(eps=0.1)),
        ("GLE", GLE(kT=1.0, mass=1.0, gamma=[1.0, 2.0], c=[1.0, 2.0]))
    ]

    fwd_times = []
    bwd_times = []
    names = []

    # Simple Harmonic force
    def force_fn(x):
        return -x
    def grad_fn(x):
        return x

    x0 = torch.randn(n_particles, dim, device=device, requires_grad=True)
    v0 = torch.randn(n_particles, dim, device=device, requires_grad=True)

    import time

    for name, integrator in integrators_list:
        integrator = integrator.to(device)
        names.append(name)

        # Helper to run integrator
        def run_int():
            if name == "ESH":
                return integrator.run(x0, None, grad_fn, n_steps=n_steps)
            elif name == "Overdamped":
                return integrator.run(x0, force_fn, dt=dt, n_steps=n_steps)
            elif name == "Verlet":
                return integrator.run(x0, v0, force_fn, dt=dt, n_steps=n_steps)
            else:
                return integrator.run(x0, None, force_fn, dt=dt, n_steps=n_steps)

        # Warmup
        try:
            run_int()
        except Exception:
            pass

        # Forward
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        out = run_int()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_fwd = time.perf_counter() - t0
        fwd_times.append(t_fwd)

        # Backward
        if isinstance(out, tuple):
            loss = out[0].sum()
        else:
            loss = out.sum()

        # Reset gradients
        if x0.grad is not None: x0.grad.zero_()
        if v0.grad is not None: v0.grad.zero_()
        for p in integrator.parameters():
            if p.grad is not None: p.grad.zero_()

        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_bwd = time.perf_counter() - t0
        bwd_times.append(t_bwd)

    return names, fwd_times, bwd_times

names, fwd_times, bwd_times = run_benchmark_for_plot()

# Bar plot
x = np.arange(len(names))
width = 0.35

rects1 = ax_bench.bar(x - width/2, fwd_times, width, label='Forward (Execution)', color=COLORS['Overdamped'], alpha=0.8)
rects2 = ax_bench.bar(x + width/2, bwd_times, width, label='Backward (Gradient)', color=COLORS['BAOAB'], alpha=0.8)

ax_bench.set_ylabel('Time (s)')
ax_bench.set_title('Performance Benchmark (100 steps, batch=256, dim=64)', fontweight='bold')
ax_bench.set_xticks(x)
ax_bench.set_xticklabels(names)
ax_bench.legend()
ax_bench.set_axisbelow(True)
ax_bench.grid(axis='y', alpha=0.3)

plt.savefig(os.path.join(assets_dir, "integrators.png"), dpi=150,
            bbox_inches='tight', facecolor='#FAFBFC')
import time
import tracemalloc
import gc

def benchmark_integrators():
    print("\n" + "="*60)
    print(f"{'Integrator':<20} | {'Time (s)':<10} | {'Steps/sec':<10} | {'Peak Mem (MB)':<12}")
    print("-" * 60)

    # Benchmark setup
    dim = 100
    n_particles = 1000
    n_steps = 1000
    dt = 0.01
    device = torch.device('cpu')

    # Simple Harmonic force for benchmarking
    def force_fn(x):
        return -x

    def grad_fn(x):
        return x

    x0 = torch.randn(n_particles, dim, device=device)
    v0 = torch.randn(n_particles, dim, device=device)

    integrators = [
        ("OverdampedLangevin", OverdampedLangevin(gamma=1.0, kT=1.0)),
        ("BAOAB", BAOAB(gamma=1.0, kT=1.0, mass=1.0)),
        ("VelocityVerlet", VelocityVerlet(mass=1.0)),
        ("NoseHoover", NoseHoover(kT=1.0, mass=1.0, Q=1.0)),
        ("NoseHooverChain", NoseHooverChain(kT=1.0, mass=1.0, Q=1.0, n_chain=2)),
        ("ESH", ESH(eps=0.1)),
        ("GLE", GLE(kT=1.0, mass=1.0, gamma=[1.0, 2.0], c=[1.0, 2.0]))
    ]

    for name, integrator in integrators:
        integrator = integrator.to(device)

        # Helper to run integrator with correct signature
        def run_int(steps, use_warmup_slice=False):
            current_x0 = x0[:10] if use_warmup_slice else x0
            current_v0 = v0[:10] if use_warmup_slice else v0

            if name == "ESH":
                integrator.run(current_x0, None, grad_fn, n_steps=steps)
            elif name == "OverdampedLangevin":
                integrator.run(current_x0, force_fn, dt=dt, n_steps=steps)
            elif name == "VelocityVerlet":
                # VelocityVerlet requires explicit v0
                integrator.run(current_x0, current_v0, force_fn, dt=dt, n_steps=steps)
            else:
                integrator.run(current_x0, None, force_fn, dt=dt, n_steps=steps)

        # Warmup
        try:
            run_int(10, use_warmup_slice=True)
        except Exception as e:
            print(f"Failed warmup for {name}: {e}")
            continue

        # Reset memory tracking
        tracemalloc.start()
        tracemalloc.clear_traces()
        start_mem = tracemalloc.get_traced_memory()[0]

        start_time = time.perf_counter()

        # Run benchmark
        run_int(n_steps, use_warmup_slice=False)

        end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = end_time - start_time
        # Metric: particles * steps / second
        throughput = (n_steps * n_particles) / duration
        mem_usage = (peak_mem - start_mem) / 1024 / 1024  # MB

        print(f"{name:<20} | {duration:<10.4f} | {throughput:<10.0f} | {mem_usage:<12.2f}")

    print("="*60 + "\n")

benchmark_integrators()
