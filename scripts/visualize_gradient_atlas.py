#!/usr/bin/env python3
"""
Gradient Method Atlas: Visual Schematic
========================================

Generates a compact schematic illustrating the landscape of gradient estimation
methods as discussed in `uni-diffsim/research-plan.md`.

Key conceptual axes visualized:
1. **Path vs Ensemble**: Single trajectory memory vs statistical forgetting
2. **Forward vs Backward**: Sensitivity propagation direction
3. **SDE vs ODE**: Stochastic noise vs deterministic thermostat
4. **BPTT vs Likelihood Ratio**: Backprop (Jacobian chain) vs path score accumulation
5. **Fine vs Coarse**: Sample-level (ML) vs observable-level matching

Visual style: Particles bouncing in a double-well potential to convey
"molecular simulation" rather than generic diffusion plots.

Minimal axes/ticks to focus on concepts. Horizontal layout for compactness.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uni_diffsim.plotting import apply_style, COLORS, LW
from uni_diffsim.integrators import OverdampedLangevin, NoseHoover
from uni_diffsim.potentials import Potential
from uni_diffsim.device import get_device

# =============================================================================
# Configuration & Hardware Acceleration
# =============================================================================

DEVICE = get_device()
torch.manual_seed(42)
np.random.seed(42)

apply_style()

# Extended palette
ATLAS_COLORS = {
    **COLORS,
    'path_single': '#2E3440',
    'sensitivity': '#EBCB8B',
    'ensemble_base': '#88C0D0',
    'noise': '#B48EAD',
    'thermostat': '#D08770',
    'arrow_fwd': '#5E81AC',
    'arrow_bwd': '#BF616A',
    'equilibrium': '#A3BE8C',
    'annotation': '#3B4252',
    'fine': '#BF616A',       # Red for fine-grained (ML)
    'coarse': '#5E81AC',     # Blue for coarse-grained (observable)
    'data': '#EBCB8B',       # Yellow for data points
    'potential': '#E5E9F0',  # Light gray for potential well fill
    'potential_line': '#4C566A',  # Darker gray for potential outline
    'jacobian': '#D08770',   # Orange for Jacobian vectors (BPTT)
    'lr_score': '#B48EAD',   # Purple for likelihood ratio scores
    'particle': '#2E3440',   # Dark for particle markers
}


# =============================================================================
# Double-Well Potential
# =============================================================================

class DoubleWell(Potential):
    """Double-well potential: U(x) = a*(x^2 - b)^2"""
    def __init__(self, a=1.0, b=0.5):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a, device=DEVICE))
        self.b = torch.nn.Parameter(torch.tensor(b, device=DEVICE))
    
    def energy(self, x):
        return self.a * (x**2 - self.b)**2
    
POTENTIAL = DoubleWell(a=1.2, b=0.5).to(DEVICE)

def double_well_potential_np(x, a=0.8, b=0.5):
    """Legacy helper for plotting."""
    return a * (x**2 - b)**2

# =============================================================================
# Unified Simulation Helpers
# =============================================================================

def run_sim(integrator, x0, n_steps, dt, store_every=1):
    """Unified runner for all integrators on DEVICE."""
    torch.manual_seed(100)
    x0 = x0.to(DEVICE)
    if isinstance(integrator, NoseHoover):
        traj_x, traj_v = integrator.run(x0, v0=None, force_fn=POTENTIAL.force, 
                                       dt=dt, n_steps=n_steps, store_every=store_every)
        return traj_x.detach().cpu().numpy(), traj_v.detach().cpu().numpy()
    else:
        traj_x = integrator.run(x0, force_fn=POTENTIAL.force, 
                               dt=dt, n_steps=n_steps, store_every=store_every)
        return traj_x.detach().cpu().numpy()

# =============================================================================
# Exemplar Trajectories - Single try, long simulation, fixed seed
# =============================================================================

# Global cache for the exemplar trajectory
_EXEMPLAR_TRAJECTORY = None
_EXEMPLAR_ODE_TRAJECTORY = None

@torch.no_grad()
def get_exemplar_trajectory():
    """Shared SDE trajectory (cached)."""
    n_steps, dt = 1000, 0.01
    od = OverdampedLangevin(gamma=1.0, kT=0.15).to(DEVICE)
    x0 = torch.tensor([-0.7], device=DEVICE)
    traj = run_sim(od, x0, n_steps, dt)
    return (np.linspace(0, 1, n_steps+1), traj[:, 0])

@torch.no_grad()
def get_exemplar_ode_trajectory():
    """Shared ODE trajectory (cached)."""
    n_steps, dt = 1000, 0.04
    nh = NoseHoover(kT=0.15, mass=1.0, Q=2.5).to(DEVICE)
    x0 = torch.tensor([-0.7], device=DEVICE)
    traj_x, traj_v = run_sim(nh, x0, n_steps, dt)
    return (np.linspace(0, 1, n_steps+1), traj_x[:, 0], traj_v[:, 0])


def clean_axis(ax):
    """Remove ticks and spines for a clean conceptual look."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('#FAFBFC')


def draw_energy_background(ax, barrier_height=0.6, alpha=0.6, t_range=(0, 4.5), y_range=(-1.1, 1.1)):
    """Draw energy landscape as colored background gradient."""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a 2D grid for the background
    t_grid = np.linspace(t_range[0], t_range[1], 100)
    y_grid = np.linspace(y_range[0], y_range[1], 200)
    T, Y = np.meshgrid(t_grid, y_grid)
    
    # Compute potential energy at each y position (independent of t)
    U = double_well_potential_np(Y, a=barrier_height, b=0.5)
    
    # Normalize so that wells are at 0 and barrier peak is at 1
    U_min = U.min()
    U_max = U.max()
    U_norm = (U - U_min) / (U_max - U_min + 1e-10)
    
    # Apply a power transform to increase contrast (make wells more distinct)
    U_contrast = U_norm ** 0.5  # Compress high values, spread low values
    
    # Draw the background
    ax.imshow(U_contrast, extent=[t_range[0], t_range[1], y_range[0], y_range[1]], 
              origin='lower', aspect='auto', cmap="Greys", alpha=0.3, zorder=0)
    
    # # Add strong contour lines to clearly show the double-well structure
    # contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    # ax.contour(T, Y, U_norm, levels=contour_levels, colors='#1565C0',
    #            alpha=0.4, linewidths=0.8, zorder=1, )


def draw_flow_arrow(ax, t, x, idx, direction='forward', color='black', size=10):
    """Draw an arrow indicating gradient flow direction."""
    if idx >= len(t) - 5 or idx < 5:
        return
    if direction == 'forward':
        dx, dy = t[idx + 4] - t[idx], x[idx + 4] - x[idx]
    else:
        dx, dy = t[idx - 4] - t[idx], x[idx - 4] - x[idx]
    ax.annotate('', xy=(t[idx] + dx * 0.8, x[idx] + dy * 0.8),
                xytext=(t[idx], x[idx]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8, mutation_scale=size))


# =============================================================================
# Row 1: PATH vs ENSEMBLE
# =============================================================================

def plot_path(ax):
    """Single path - remembers everything. Pathwise/reparameterization gradient.
    
    Pathwise: E[∇_x O · ∂x_T/∂θ] - explicit Jacobian Φ(T,t)
    Variance ~ exp(2λT) due to Lyapunov instability
    
    Visual: Single particle trajectory in double-well with sensitivity chain.
    Shows barrier crossing events with lower barrier and longer simulation.
    """
    # Get the exemplar trajectory (shared across panels)
    t, x = get_exemplar_trajectory()
    t_range = (t[0], t[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.55, alpha=0.3, t_range=t_range)
    
    # Plot the trajectory
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.9, zorder=5)
    
    # Mark sensitivity chain - Jacobian Φ(T,t) propagates through history
    n = len(t)
    events = [int(n*0.12), int(n*0.35), int(n*0.55), int(n*0.80)]
    for ev in events:
        ax.scatter(t[ev], x[ev], s=50, color=ATLAS_COLORS['arrow_bwd'], 
                  edgecolor='white', linewidth=1, zorder=10)
    for i in range(len(events) - 1):
        ax.annotate('', xy=(t[events[i+1]], x[events[i+1]]),
                    xytext=(t[events[i]], x[events[i]]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['sensitivity'],
                                   lw=1.5, ls='--', connectionstyle='arc3,rad=0.15'))
    
    ax.set_title('PATH', fontsize=11, fontweight='bold', color=ATLAS_COLORS['path_single'], pad=4)
    ax.text(0.5, 0.02, '"remembers"', transform=ax.transAxes, ha='center', fontsize=9,
            style='italic', color=ATLAS_COLORS['annotation'])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


def plot_ensemble(ax):
    """Ensemble - forgets history. Uses equilibrium distribution only."""
    # Get reference trajectory for time axis
    t_ref, _ = get_exemplar_trajectory()
    
    # Generate ensemble of trajectories in BATCH
    n_paths = 8
    dt = 0.012
    x0_values = torch.tensor([-0.75 if i % 2 == 0 else 0.75 for i in range(n_paths)], device=DEVICE)
    od = OverdampedLangevin(gamma=1.0, kT=0.15).to(DEVICE)
    x_batch = run_sim(od, x0_values, len(t_ref)-1, dt)
    t = np.linspace(0, (len(t_ref)-1)*dt, len(t_ref))
    t_end = t[-1]
    
    # Plot all trajectories
    for i in range(n_paths):
        x = x_batch[:, i]
        color = plt.cm.Blues(0.3 + 0.4 * (x[-1] + 1) / 2)
        ax.plot(t, x, color=color, alpha=0.25, lw=0.8, zorder=2)
    
    # Equilibrium distribution - bimodal for double-well
    y_dist = np.linspace(-1.05, 1.05, 100)
    eq_dist = 0.5 * np.exp(-8 * (y_dist - 0.7)**2) + 0.5 * np.exp(-8 * (y_dist + 0.7)**2)
    eq_dist = eq_dist / eq_dist.max() * 0.7
    
    # Draw distribution at the end
    ax.fill_betweenx(y_dist, t_end, t_end + eq_dist, color=ATLAS_COLORS['equilibrium'], alpha=0.5, zorder=3)
    ax.plot(t_end + eq_dist, y_dist, color=ATLAS_COLORS['equilibrium'], lw=1.8, zorder=4)
    ax.text(t_end + 0.45, 0, '$\\pi$', fontsize=10, color=ATLAS_COLORS['equilibrium'], fontweight='bold')
    
    t_range = (t[0], t_end + 0.85)
    draw_energy_background(ax, barrier_height=0.5, alpha=0.3, t_range=t_range)
    
    ax.set_title('ENSEMBLE', fontsize=11, fontweight='bold', color=ATLAS_COLORS['ensemble_base'], pad=4)
    ax.text(0.5, 0.02, '"forgets"', transform=ax.transAxes, ha='center', fontsize=9,
            style='italic', color=ATLAS_COLORS['annotation'])
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


# =============================================================================
# Row 2: FORWARD vs BACKWARD
# =============================================================================

def plot_forward(ax):
    """Forward sensitivity propagation - SDE forward integration.
    
    Visual: Particle in double-well with forward-pointing arrows BETWEEN adjacent
    checkpoints showing the forward SDE integration flow from t=0 to t=T.
    """
    # Get the exemplar trajectory (shared across panels)
    t, x = get_exemplar_trajectory()
    t_range = (t[0], t[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.55, alpha=0.3, t_range=t_range)
    
    # Plot the trajectory
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.7, zorder=5)
    
    # =====================================================================
    # KEY: Forward arrows BETWEEN adjacent checkpoints (SDE integration flow)
    # Shows x_{t} → x_{t+1} forward propagation
    # =====================================================================
    n = len(t)
    checkpoint_indices = [int(n*0.10), int(n*0.30), int(n*0.50), int(n*0.70), int(n*0.90)]
    
    # Mark all checkpoints
    for idx in checkpoint_indices:
        ax.scatter(t[idx], x[idx], s=40, color=ATLAS_COLORS['arrow_fwd'], 
                  edgecolor='white', linewidth=1, zorder=10)
    
    # Draw forward arrows BETWEEN adjacent checkpoints
    for i in range(len(checkpoint_indices) - 1):
        idx_from = checkpoint_indices[i]
        idx_to = checkpoint_indices[i + 1]
        
        # Arrow from checkpoint i to checkpoint i+1 (forward flow)
        ax.annotate('', xy=(t[idx_to], x[idx_to]), xytext=(t[idx_from], x[idx_from]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['arrow_fwd'], 
                                   lw=1.8, alpha=0.8, mutation_scale=12,
                                   connectionstyle='arc3,rad=0.1'))
    
    # Add labels for first and last checkpoints
    ax.text(t[checkpoint_indices[0]], x[checkpoint_indices[0]] - 0.18, '$x_0$', 
            fontsize=8, ha='center', color=ATLAS_COLORS['arrow_fwd'], fontweight='bold')
    ax.text(t[checkpoint_indices[-1]], x[checkpoint_indices[-1]] + 0.18, '$x_T$', 
            fontsize=8, ha='center', color=ATLAS_COLORS['arrow_fwd'], fontweight='bold')
    
    ax.set_title('FORWARD', fontsize=11, fontweight='bold', color=ATLAS_COLORS['arrow_fwd'], pad=4)
    ax.text(0.5, 0.02, '$x_t \\to x_{t+1}$ (SDE)', transform=ax.transAxes, ha='center', fontsize=9,
            color=ATLAS_COLORS['arrow_fwd'])
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


def plot_backward(ax):
    """Backward adjoint flow - Jacobian propagation.
    
    Visual: Particle in double-well with REVERSE arrows BETWEEN adjacent checkpoints
    showing Jacobian backpropagation from T to 0.
    """
    # Get the exemplar trajectory (shared across panels)
    t, x = get_exemplar_trajectory()
    t_range = (t[0], t[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.55, alpha=0.3, t_range=t_range)
    
    # Plot the trajectory
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.7, zorder=5)
    
    # =====================================================================
    # KEY: Reverse arrows BETWEEN adjacent checkpoints (Jacobian backprop)
    # Shows ∂L/∂x_{t} ← ∂L/∂x_{t+1} backward propagation
    # =====================================================================
    n = len(t)
    checkpoint_indices = [int(n*0.10), int(n*0.30), int(n*0.50), int(n*0.70), int(n*0.90)]
    
    # Mark all checkpoints
    for idx in checkpoint_indices:
        ax.scatter(t[idx], x[idx], s=40, color=ATLAS_COLORS['arrow_bwd'], 
                  edgecolor='white', linewidth=1, zorder=10)
    
    # Draw REVERSE arrows BETWEEN adjacent checkpoints (backward Jacobian flow)
    for i in range(len(checkpoint_indices) - 1, 0, -1):
        idx_from = checkpoint_indices[i]      # Later time (gradient source)
        idx_to = checkpoint_indices[i - 1]    # Earlier time (gradient target)
        
        # Arrow from checkpoint i to checkpoint i-1 (backward Jacobian flow)
        ax.annotate('', xy=(t[idx_to], x[idx_to]), xytext=(t[idx_from], x[idx_from]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['arrow_bwd'], 
                                   lw=1.8, alpha=0.8, mutation_scale=12,
                                   connectionstyle='arc3,rad=-0.1'))
    
    # Add labels for first and last checkpoints
    ax.text(t[checkpoint_indices[-1]], x[checkpoint_indices[-1]] + 0.18, '$\\nabla_{x_T}\\mathcal{L}$', 
            fontsize=7, ha='center', color=ATLAS_COLORS['arrow_bwd'], fontweight='bold')
    ax.text(t[checkpoint_indices[0]], x[checkpoint_indices[0]] - 0.18, '$\\nabla_{x_0}\\mathcal{L}$', 
            fontsize=7, ha='center', color=ATLAS_COLORS['arrow_bwd'], fontweight='bold')
    
    ax.set_title('BACKWARD', fontsize=11, fontweight='bold', color=ATLAS_COLORS['arrow_bwd'], pad=4)
    ax.text(0.5, 0.02, '$\\nabla_{x_t} \\leftarrow \\nabla_{x_{t+1}}$ (Jacobian)', transform=ax.transAxes, ha='center', fontsize=8,
            color=ATLAS_COLORS['arrow_bwd'])
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


# =============================================================================
# Row 3: SDE vs ODE
# =============================================================================

def plot_sde(ax):
    """SDE with external noise.
    
    Visual: Particle in double-well with random noise kicks (dW_t) shown as
    arrows indicating stochastic perturbations.
    """
    # Get the exemplar trajectory (shared across panels)
    t, x = get_exemplar_trajectory()
    t_range = (t[0], t[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.55, alpha=0.3, t_range=t_range)
    
    # Plot the trajectory
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.8, zorder=5)
    
    # Noise kicks - random perturbations shown as arrows
    np.random.seed(55)
    n = len(t)
    noise_pts = np.random.choice(range(int(n*0.1), int(n*0.85)), size=12, replace=False)
    for pt in noise_pts:
        noise_mag = np.random.randn() * 0.18
        ax.annotate('', xy=(t[pt], x[pt] + noise_mag),
                    xytext=(t[pt], x[pt]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['noise'], 
                                   lw=1.3, alpha=0.7))
    
    ax.set_title('SDE', fontsize=11, fontweight='bold', color=ATLAS_COLORS['noise'], pad=4)
    ax.text(0.5, 0.02, '$dW_t$ noise', transform=ax.transAxes, ha='center', fontsize=9,
            color=ATLAS_COLORS['noise'])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


def plot_ode(ax):
    """ODE with deterministic thermostat.
    
    Visual: Particle in double-well with Nosé-Hoover thermostat variable ξ
    shown as a coupled oscillator controlling the dynamics.
    Uses a long pre-equilibrated simulation to show proper barrier crossing.
    """
    # Get the exemplar ODE trajectory (pre-equilibrated)
    t, x, xi = get_exemplar_ode_trajectory()
    t_range = (t[0], t[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.55, alpha=0.3, t_range=t_range)
    
    # Plot the trajectory
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.2, alpha=0.85, zorder=5)
    
    # Thermostat variable ξ (scaled to fit in plot)
    xi_scaled = xi * 0.12
    ax.plot(t, xi_scaled, color=ATLAS_COLORS['thermostat'], lw=0.9, alpha=0.5, ls='--', zorder=4)
    
    # Coupling indicators between x and ξ at a few points
    n = len(t)
    for pt in [int(n*0.2), int(n*0.5), int(n*0.8)]:
        if pt < len(t):
            ax.plot([t[pt], t[pt]], [x[pt], xi_scaled[pt]], 
                   color=ATLAS_COLORS['thermostat'], ls=':', lw=1.2, alpha=0.5)
    
    ax.set_title('ODE', fontsize=11, fontweight='bold', color=ATLAS_COLORS['thermostat'], pad=4)
    ax.text(0.5, 0.02, '$\\xi$ thermostat', transform=ax.transAxes, ha='center', fontsize=9,
            color=ATLAS_COLORS['thermostat'])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


# =============================================================================
# Row 4: BPTT (Backprop) vs LIKELIHOOD RATIO (Path Score)
# =============================================================================

def plot_bptt(ax):
    """BPTT (Backpropagation Through Time) - Jacobian chain propagation.
    
    Pathwise gradient: ∂x_T/∂θ = Φ(T,0) = Π_t ∂f/∂x where Φ is the Jacobian matrix.
    The sensitivity grows exponentially: ||Φ|| ~ exp(λT) where λ is Lyapunov exponent.
    
    Visual: Show trajectory with Jacobian vectors growing along the path,
    illustrating the exponential growth that causes instability in chaotic systems.
    Shows the chain rule accumulation: Φ(T,0) = Φ(T,t_n) · Φ(t_n,t_{n-1}) · ... · Φ(t_1,0)
    """
    # Get the exemplar trajectory (shared across panels)
    t, x = get_exemplar_trajectory()
    t_range = (t[0], t[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.55, alpha=0.3, t_range=t_range)
    
    # Plot the trajectory
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.7, zorder=5)
    
    # Show Jacobian vectors GROWING along the path (exponential growth)
    # These represent ∂x_t/∂θ - the sensitivity to parameter changes
    n = len(t)
    jacobian_pts = [int(n*0.12), int(n*0.30), int(n*0.48), int(n*0.65), int(n*0.82)]
    base_size = 0.06
    
    for i, pt in enumerate(jacobian_pts):
        # Jacobian grows exponentially: size ~ exp(λt)
        growth_factor = np.exp(0.45 * i)  # Simulated exponential growth
        vec_size = base_size * growth_factor
        
        # Tangent direction (approximate)
        if pt < len(t) - 5:
            tangent_x = t[pt+5] - t[pt]
            tangent_y = x[pt+5] - x[pt]
            norm = np.sqrt(tangent_x**2 + tangent_y**2) + 1e-6
            # Perpendicular direction
            perp_x = -tangent_y / norm * vec_size * 0.3
            perp_y = tangent_x / norm * vec_size * 0.8
        else:
            perp_x, perp_y = 0, vec_size * 0.8
        
        # Draw growing Jacobian vector
        ax.annotate('', xy=(t[pt] + perp_x, x[pt] + perp_y),
                    xytext=(t[pt], x[pt]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['jacobian'], 
                                   lw=1.5 + 0.4*i, alpha=0.8, mutation_scale=8+i*2))
        
        # Mark the point
        ax.scatter(t[pt], x[pt], s=30 + 12*i, color=ATLAS_COLORS['jacobian'], 
                  edgecolor='white', linewidth=0.8, zorder=10)
    
    # Label showing Jacobian chain rule
    ax.text(t[jacobian_pts[-1]] - 0.15, x[jacobian_pts[-1]] + 0.35, 
            '$\\Phi(T,0)$', fontsize=8, color=ATLAS_COLORS['jacobian'], fontweight='bold')
    
    # Formula annotation showing chain rule accumulation (positioned to match PATH LR)
    ax.text(0.02, 0.82, '$\\Phi = \\prod_t \\frac{\\partial f}{\\partial x}$', 
            transform=ax.transAxes, fontsize=7, color=ATLAS_COLORS['jacobian'],
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax.set_title('BPTT (Backprop)', fontsize=11, fontweight='bold', 
                 color=ATLAS_COLORS['jacobian'], pad=4)
    ax.text(0.5, 0.02, 'var $\\sim e^{2\\lambda T}$', 
            transform=ax.transAxes, ha='center', fontsize=8, 
            color=ATLAS_COLORS['jacobian'], style='italic')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


def plot_likelihood_ratio(ax):
    """Likelihood Ratio (Path Score) - score accumulation along trajectory.
    
    Path gradient: ∇_θ log p(τ|θ) = Σ_t δℓ_t where δℓ_t are local score increments.
    Variance grows linearly: Var ~ T (much better than exp(2λT)!)
    
    δℓ_t = (1/σ²) ∇_θ b(x_t) · dW_t  (Girsanov-Cameron-Martin)
    
    Visual: Show trajectory with score increments accumulating ABOVE the trajectory,
    building up to a final sum (path integral). This shows the "path sum" concept.
    """
    # Get the exemplar trajectory (shared across panels)
    t, x = get_exemplar_trajectory()
    t_range = (t[0], t[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.55, alpha=0.3, t_range=t_range)
    
    # Plot the trajectory
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.7, zorder=5)
    
    # =====================================================================
    # KEY: Path sum ABOVE trajectory - cumulative score accumulation
    # Shows δℓ_t increments stacking up to form the total path score
    # =====================================================================
    n = len(t)
    score_pts = [int(n*0.20), int(n*0.40), int(n*0.60), int(n*0.80)]
    
    # Baseline for the accumulated score (above trajectory, within visible range)
    score_baseline = 0.35
    cumulative_height = 0.0
    increment_height = 0.12  # Each δℓ adds this much
    
    # Draw accumulating score bars ABOVE the trajectory
    for i, pt in enumerate(score_pts):
        # Draw connection from trajectory point to score accumulator
        ax.plot([t[pt], t[pt]], [x[pt], score_baseline + cumulative_height], 
               color=ATLAS_COLORS['lr_score'], ls=':', lw=0.8, alpha=0.4, zorder=6)
        
        # Mark the sample point on trajectory
        ax.scatter(t[pt], x[pt], s=25, color=ATLAS_COLORS['lr_score'], 
                  edgecolor='white', linewidth=0.5, zorder=10)
        
        # Draw the increment bar (stacking)
        bar_width = 0.035
        bar_left = t[pt] - bar_width/2
        rect = patches.Rectangle((bar_left, score_baseline + cumulative_height), 
                                  bar_width, increment_height,
                                  facecolor=ATLAS_COLORS['lr_score'], 
                                  edgecolor='white', linewidth=0.5,
                                  alpha=0.7, zorder=8)
        ax.add_patch(rect)
        
        cumulative_height += increment_height
    
    # Final accumulated sum marker (at the top of the stack)
    final_score_y = score_baseline + cumulative_height
    ax.scatter(t[score_pts[-1]], final_score_y + 0.03, s=50, 
              color=ATLAS_COLORS['lr_score'], marker='s',
              edgecolor='white', linewidth=1, zorder=15)
    
    # Labels (positioned within bounds)
    ax.text(t[score_pts[0]] + 0.025, score_baseline + increment_height * 0.5, '$\\delta\\ell$', 
            fontsize=7, color=ATLAS_COLORS['lr_score'], fontweight='bold', va='center')
    ax.text(t[score_pts[-1]] + 0.025, final_score_y, '$\\sum\\delta\\ell$', 
            fontsize=7, color=ATLAS_COLORS['lr_score'], fontweight='bold', va='center')
    
    # Formula annotation (positioned to match BPTT)
    ax.text(0.02, 0.82, '$\\delta\\ell_t \\propto \\nabla_\\theta b \\cdot dW$', 
            transform=ax.transAxes, fontsize=7, color=ATLAS_COLORS['lr_score'],
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax.set_title('PATH LR', fontsize=11, fontweight='bold', 
                 color=ATLAS_COLORS['lr_score'], pad=4)
    ax.text(0.5, 0.02, 'var $\\sim T$', 
            transform=ax.transAxes, ha='center', fontsize=8, 
            color=ATLAS_COLORS['lr_score'], style='italic')
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


# =============================================================================
# Row 5: FINE-GRAINED (Data Likelihood) vs COARSE-GRAINED (Observable) LOSS
# =============================================================================
# 
# KEY INSIGHT: Both are instances of the SAME gradient formula:
#     ∇_θ L = -β Cov_π(O, ∇_θ U)
# 
# The difference is the LOSS FUNCTION (choice of what to match):
#   - CD Loss: L = E_data[E] - E_model[E] → match full distribution
#   - Observable Loss: L = (⟨O⟩ - target)² → match statistics
# 
# This is the "loss function perspective" on differentiable simulation.
# Both losses lead to the same covariance gradient structure!
# =============================================================================

def plot_data_likelihood(ax):
    """Contrastive Divergence Loss: L = E_data[E] - E_model[E]
    
    The CD loss directly compares energies at data vs model samples.
    This is maximum likelihood training of energy-based models.
    
    Gradient: ∇_θ L = E_data[∇_θE] - E_model[∇_θE]
    
    Visual: Show data points (×) and model samples (○) with their energies,
    emphasizing the energy difference that drives learning.
    """
    # Get reference trajectory for time axis
    t_ref, _ = get_exemplar_trajectory()
    
    # Generate simulation trajectories (model samples) in BATCH
    n_traj = 6
    dt = 0.012
    x0_values = torch.tensor([-0.75 if i % 2 == 0 else 0.75 for i in range(n_traj)], device=DEVICE)
    od = OverdampedLangevin(gamma=1.0, kT=0.15).to(DEVICE)
    x_batch = run_sim(od, x0_values, len(t_ref)-1, dt)
    t = np.linspace(0, (len(t_ref)-1)*dt, len(t_ref))
    t_end = t[-1]
    
    # Draw energy background FIRST
    t_range = (t_ref[0], t_end + 0.15)
    draw_energy_background(ax, barrier_height=0.5, alpha=0.25, t_range=t_range)
    
    # Plot model trajectories (lighter, in background)
    for i in range(n_traj):
        ax.plot(t, x_batch[:, i], color=ATLAS_COLORS['fine'], alpha=0.35, lw=0.8, zorder=3)
    
    # =====================================================================
    # KEY: Data points (×) vs Model samples (○)
    # CD Loss = E_data[E(x)] - E_model[E(x)]
    # =====================================================================
    time_slices = [0.15, 0.40, 0.65, 0.90]  # 4 time slices
    
    # Generate "data" samples at each time slice
    np.random.seed(123)
    data_per_slice = [
        np.array([-0.65, -0.55, 0.50]),       # t1: data near left well + one right
        np.array([-0.45, 0.35, 0.60]),        # t2: spreading toward both wells
        np.array([-0.70, 0.25, 0.55, 0.70]),  # t3: more in right well
        np.array([-0.60, 0.50, 0.65]),        # t4: equilibrium-like
    ]
    
    for slice_idx, (frac, data_samples) in enumerate(zip(time_slices, data_per_slice)):
        t_idx = int(frac * (len(t) - 1))
        t_slice = t[t_idx]
        
        # Draw DATA samples (×) - these have E_data
        for y_data in data_samples:
            ax.scatter(t_slice, y_data, s=45, color="black", linewidth=1.2, zorder=12, marker='x')
    
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('DATA LIKELIHOOD', fontsize=11, fontweight='bold', color=ATLAS_COLORS['fine'], pad=4)
    ax.text(0.5, 0.02, '$\\mathcal{L} = \\mathbb{E}_{\\mathrm{data}}[\\nabla U] - \\mathbb{E}_{\\mathrm{model}}[\\nabla U]$', 
            transform=ax.transAxes, ha='center', fontsize=7, color=ATLAS_COLORS['fine'])
    clean_axis(ax)


def plot_observable(ax):
    """Observable Loss: L = (⟨O⟩ - target)²
    
    Match a statistic (observable) rather than the full distribution.
    This is coarse-grained matching - only care about specific moments.
    
    Gradient: ∇_θ L = 2(⟨O⟩ - target) · (-β) Cov_π(O, ∇_θU)
    
    Visual: Ensemble with mean trajectory, showing the gap to target.
    """
    # Get reference trajectory for time axis
    t_ref, _ = get_exemplar_trajectory()
    t_range = (t_ref[0], t_ref[-1])
    
    # Draw energy landscape as colored background
    draw_energy_background(ax, barrier_height=0.5, alpha=0.3, t_range=t_range)
    
    # Generate ensemble in BATCH
    n_paths = 15
    dt = 0.012
    # Start all from one side to show relaxation of mean
    x0_values = torch.tensor([-0.8 for _ in range(n_paths)], device=DEVICE)
    od = OverdampedLangevin(gamma=1.0, kT=0.15).to(DEVICE)
    x_batch = run_sim(od, x0_values, len(t_ref)-1, dt)
    t = np.linspace(0, 1, len(t_ref))
    t_end = t[-1]
    
    # Plot all trajectories (faded - individual paths don't matter)
    for i in range(n_paths):
        ax.plot(t, x_batch[:, i], color=ATLAS_COLORS['coarse'], alpha=0.20, lw=0.6, zorder=2)
    
    # =====================================================================
    # KEY: Only the OBSERVABLE ⟨x⟩_t matters, not individual samples
    # Loss = (⟨x⟩_T - target)²
    # =====================================================================
    mean_traj = np.mean(x_batch, axis=1)
    
    # Plot the observable curve (thick, prominent - THIS is what we're matching)
    ax.plot(t, mean_traj, color=ATLAS_COLORS['coarse'], lw=2.8, zorder=10, alpha=0.95)
    
    # Label the observable
    mid_idx = len(t) // 3
    ax.text(t[mid_idx], mean_traj[mid_idx] + 0.20, '$\\langle x \\rangle_t$', 
            fontsize=9, color=ATLAS_COLORS['coarse'], fontweight='bold', ha='center', zorder=15)
    
    # Show target value as dashed line
    target_mean = 0.0  # Equilibrium mean for symmetric double-well
    ax.axhline(y=target_mean, color=ATLAS_COLORS['data'], ls='--', lw=1.5, alpha=0.7, zorder=8)
    ax.text(t_end * 0.85, target_mean + 0.12, '$\\langle x \\rangle^*$', 
            fontsize=8, color=ATLAS_COLORS['data'], fontweight='bold', va='bottom', ha='center')
    
    ax.set_title('OBSERVABLE', fontsize=11, fontweight='bold', color=ATLAS_COLORS['coarse'], pad=4)
    ax.text(0.5, 0.02, '$\\mathcal{L} = (\\langle x \\rangle_T - \\langle x \\rangle^*)^2$', 
            transform=ax.transAxes, ha='center', fontsize=8, color=ATLAS_COLORS['coarse'])
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(-1.1, 1.1)
    clean_axis(ax)


# =============================================================================
# Main Figure Assembly - Horizontal Layout (5 columns x 2 rows)
# =============================================================================

def main():
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('#FAFBFC')
    
    # Create 2x5 grid (2 rows, 5 columns of comparisons)
    gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.12,
                          left=0.03, right=0.98, top=0.82, bottom=0.08)
    
    # Column labels
    col_info = [
        ('I', 'Path vs Ensemble'),
        ('II', 'Forward vs Backward'),
        ('III', 'SDE vs ODE'),
        ('IV', 'BPTT vs Likelihood Ratio'),
        ('V', 'Data Likelihood vs Observable'),  # Both use same gradient, different O
    ]
    
    # Plot panels - top row (first of each pair), bottom row (second of each pair)
    ax_path = fig.add_subplot(gs[0, 0])
    ax_ensemble = fig.add_subplot(gs[1, 0])
    plot_path(ax_path)
    plot_ensemble(ax_ensemble)
    
    ax_fwd = fig.add_subplot(gs[0, 1])
    ax_bwd = fig.add_subplot(gs[1, 1])
    plot_forward(ax_fwd)
    plot_backward(ax_bwd)
    
    ax_sde = fig.add_subplot(gs[0, 2])
    ax_ode = fig.add_subplot(gs[1, 2])
    plot_sde(ax_sde)
    plot_ode(ax_ode)
    
    # Column IV: BPTT vs Likelihood Ratio
    ax_bptt = fig.add_subplot(gs[0, 3])
    ax_lr = fig.add_subplot(gs[1, 3])
    plot_bptt(ax_bptt)
    plot_likelihood_ratio(ax_lr)
    
    # Column V: Data Likelihood vs Observable
    ax_data_likelihood = fig.add_subplot(gs[0, 4])
    ax_observable = fig.add_subplot(gs[1, 4])
    plot_data_likelihood(ax_data_likelihood)
    plot_observable(ax_observable)
    
    # Column labels on top
    for i, (num, label) in enumerate(col_info):
        x_pos = 0.11 + i * 0.19
        fig.text(x_pos, 0.89, f'{num}. {label}', fontsize=9, fontweight='bold',
                color=ATLAS_COLORS['annotation'], ha='center')
    
    # "vs" labels between rows (in the middle of each column)
    # Skip column I (index 0) since it has the IBP annotation there
    for i in range(1, 5):
        x_pos = 0.11 + i * 0.19
        fig.text(x_pos, 0.47, 'vs', fontsize=9, color=ATLAS_COLORS['gray'],
                ha='center', va='center', style='italic')
    
    # ==========================================================================
    # Column I: Add IBP equivalence annotation between PATH and ENSEMBLE
    # This is the key mathematical insight: Malliavin IBP shows they're equivalent
    # ==========================================================================
    ibp_color = '#8B5CF6'  # Purple for the IBP connection
    
    # Position for the IBP annotation (between the two panels of column I)
    x_ibp = 0.11
    
    # Top formula (PATH): E[∇_x O · ∂x_T/∂θ] - the pathwise/reparameterization gradient
    fig.text(x_ibp - 0.05, 0.54, '$\\mathbb{E}[\\nabla_x \\mathcal{O} \\cdot \\partial_\\theta x_T]$', 
             fontsize=9, ha='center', va='bottom', color=ATLAS_COLORS['path_single'],
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                                          edgecolor=ATLAS_COLORS['path_single'], alpha=0.8, lw=0.5))
    
    # Large equivalence symbol with "Malliavin IBP" label
    fig.text(x_ibp - 0.05, 0.47, '≡', fontsize=14, ha='center', va='center', 
             color=ibp_color, fontweight='bold')
    fig.text(x_ibp + 0.05, 0.47, 'Malliavin\n   IBP', fontsize=8, ha='left', va='center', 
             color=ibp_color, fontweight='bold', linespacing=0.9)
    
    # Bottom formula (ENSEMBLE): E[O · ∇_θ log p(τ)] - the score/REINFORCE gradient
    fig.text(x_ibp - 0.05, 0.40, '$\\mathbb{E}[\\mathcal{O} \\cdot \\nabla_\\theta \\log p(\\tau)]$', 
             fontsize=9, ha='center', va='top', color=ATLAS_COLORS['ensemble_base'],
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                                          edgecolor=ATLAS_COLORS['ensemble_base'], alpha=0.8, lw=0.5))
    
    # Variance comparison - the KEY practical difference!
    # PATH has exponential variance (Lyapunov), ENSEMBLE has linear variance (Itô)
    # Position these to the right of the formulas, not overlapping
    fig.text(x_ibp + 0.065, 0.545, 'var $\\sim e^{2\\lambda T}$', 
             fontsize=8.5, ha='left', va='center', color=ATLAS_COLORS['arrow_bwd'], 
             style='italic')
    fig.text(x_ibp + 0.065, 0.395, 'var $\\sim T$', 
             fontsize=8.5, ha='left', va='center', color=ATLAS_COLORS['equilibrium'], 
             style='italic')
    
    # ==========================================================================
    # Column V: Add "Loss Function" annotation between DATA LIKELIHOOD and OBSERVABLE
    # KEY INSIGHT: Both are different LOSS choices, same underlying gradient structure!
    # CD Loss: e_data - e_model  |  Observable Loss: (⟨O⟩ - target)²
    # ==========================================================================
    loss_color = '#059669'  # Emerald green for the loss connection
    x_loss = 0.11 + 4 * 0.19  # Column V position
    
    # Top: CD Loss formula
    fig.text(x_loss - 0.06, 0.54, '$\\mathcal{L}_\\mathrm{CD} = E_\\mathrm{data} - E_\\mathrm{model}$', 
             fontsize=7.5, ha='center', va='bottom', color=ATLAS_COLORS['fine'],
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                                          edgecolor=ATLAS_COLORS['fine'], alpha=0.8, lw=0.5))
    
    # "vs" with "choice of L" label - emphasizing it's about LOSS choice
    # fig.text(x_loss - 0.06, 0.47, 'vs', fontsize=9, ha='center', va='center', 
    #          color=loss_color, fontweight='bold', style='italic')
    # fig.text(x_loss + 0.04, 0.47, 'choice\nof $\\mathcal{L}$', fontsize=7.5, ha='left', va='center', 
    #          color=loss_color, fontweight='bold', linespacing=0.9)
    
    # Bottom: Observable Loss formula
    fig.text(x_loss - 0.06, 0.40, '$\\mathcal{L}_\\mathrm{obs} = (\\langle\\mathcal{O}\\rangle - \\mathcal{O}^*)^2$', 
             fontsize=7.5, ha='center', va='top', color=ATLAS_COLORS['coarse'],
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                                          edgecolor=ATLAS_COLORS['coarse'], alpha=0.8, lw=0.5))
    
    # Main title
    fig.suptitle('The Gradient Method Atlas', fontsize=15, fontweight='bold',
                y=0.97, color=ATLAS_COLORS['annotation'])
    
    # Footer
    fig.text(0.5, 0.01,
             'All methods → $-\\beta\\,\\mathrm{Cov}_\\pi(\\mathcal{O}, \\nabla_\\theta U)$ at equilibrium',
             ha='center', fontsize=10, color=ATLAS_COLORS['equilibrium'], fontweight='bold')
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gradient_atlas.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gradient Atlas saved to: {output_path}")


if __name__ == "__main__":
    main()
