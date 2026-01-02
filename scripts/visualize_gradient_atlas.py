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
4. **REINFORCE vs Girsanov**: Score function vs path reweighting
5. **Fine vs Coarse**: Sample-level (ML) vs observable-level matching

Minimal axes/ticks to focus on concepts. Horizontal layout for compactness.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uni_diffsim.plotting import apply_style, COLORS, LW

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
}


def generate_sde_trajectory(n_steps=150, dt=0.05, x0=0.0, drift_scale=0.1, 
                            noise_scale=0.3, seed=None):
    """Generate an Ornstein-Uhlenbeck-like SDE trajectory."""
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, n_steps * dt, n_steps)
    x = np.zeros(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        x[i] = x[i-1] - drift_scale * x[i-1] * dt + noise_scale * np.sqrt(dt) * np.random.randn()
    return t, x


def generate_ode_trajectory(n_steps=150, dt=0.05, x0=0.0, v0=0.5, 
                            thermostat_coupling=0.5, seed=None):
    """Generate a Nosé-Hoover-like deterministic trajectory."""
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, n_steps * dt, n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    xi = np.zeros(n_steps)
    x[0], v[0] = x0, v0
    xi[0] = np.random.randn() * 0.1 if seed else 0.0
    for i in range(1, n_steps):
        force = -0.5 * x[i-1]
        xi[i] = xi[i-1] + dt * (v[i-1]**2 - 1.0) * thermostat_coupling
        v[i] = v[i-1] + dt * (force - xi[i-1] * v[i-1])
        x[i] = x[i-1] + dt * v[i]
    return t, x, xi


def clean_axis(ax):
    """Remove ticks and spines for a clean conceptual look."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('#FAFBFC')


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
    """Single path - remembers everything."""
    t, x = generate_sde_trajectory(seed=42, n_steps=120)
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=2, alpha=0.9)
    
    # Mark memory chain
    events = [20, 50, 80, 100]
    for ev in events:
        ax.scatter(t[ev], x[ev], s=40, color=ATLAS_COLORS['arrow_bwd'], zorder=10)
    for i in range(len(events) - 1):
        ax.annotate('', xy=(t[events[i+1]], x[events[i+1]]),
                    xytext=(t[events[i]], x[events[i]]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['sensitivity'],
                                   lw=1.2, ls='--', connectionstyle='arc3,rad=0.15'))
    
    ax.set_title('PATH', fontsize=10, fontweight='bold', color=ATLAS_COLORS['path_single'], pad=3)
    ax.text(0.5, 0.02, '"remembers"', transform=ax.transAxes, ha='center', fontsize=8,
            style='italic', color=ATLAS_COLORS['annotation'])
    clean_axis(ax)


def plot_ensemble(ax):
    """Ensemble - forgets history."""
    n_paths = 20
    for i in range(n_paths):
        t, x = generate_sde_trajectory(seed=100 + i, n_steps=120, noise_scale=0.35)
        weight = np.exp(-0.3 * x[-1]**2)
        color = plt.cm.Blues(0.3 + 0.5 * weight)
        ax.plot(t, x, color=color, alpha=0.3 + 0.4 * weight, lw=1)
    
    # Equilibrium distribution
    y_range = np.linspace(-2, 2, 80)
    eq_dist = np.exp(-0.5 * y_range**2)
    eq_dist = eq_dist / eq_dist.max() * 0.8
    ax.fill_betweenx(y_range, t[-1], t[-1] + eq_dist, color=ATLAS_COLORS['equilibrium'], alpha=0.4)
    ax.plot(t[-1] + eq_dist, y_range, color=ATLAS_COLORS['equilibrium'], lw=1.5)
    ax.text(t[-1] + 0.5, 0, '$\\pi$', fontsize=10, color=ATLAS_COLORS['equilibrium'], fontweight='bold')
    
    ax.set_title('ENSEMBLE', fontsize=10, fontweight='bold', color=ATLAS_COLORS['ensemble_base'], pad=3)
    ax.text(0.5, 0.02, '"forgets"', transform=ax.transAxes, ha='center', fontsize=8,
            style='italic', color=ATLAS_COLORS['annotation'])
    ax.set_xlim(t[0] - 0.3, t[-1] + 1.5)
    clean_axis(ax)


# =============================================================================
# Row 2: FORWARD vs BACKWARD
# =============================================================================

def plot_forward(ax):
    """Forward sensitivity propagation."""
    t, x = generate_sde_trajectory(seed=42, n_steps=120)
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.7)
    
    # Forward arrows only (no uncertainty tube)
    for idx in [20, 45, 70, 95]:
        draw_flow_arrow(ax, t, x, idx, 'forward', ATLAS_COLORS['arrow_fwd'])
    
    ax.set_title('FORWARD', fontsize=10, fontweight='bold', color=ATLAS_COLORS['arrow_fwd'], pad=3)
    ax.text(0.5, 0.02, '$0 \\to T$', transform=ax.transAxes, ha='center', fontsize=9,
            color=ATLAS_COLORS['arrow_fwd'])
    clean_axis(ax)


def plot_backward(ax):
    """Backward adjoint flow."""
    t, x = generate_sde_trajectory(seed=42, n_steps=120)
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.7)
    
    # Loss marker
    ax.scatter(t[-1], x[-1], s=80, color=ATLAS_COLORS['arrow_bwd'], zorder=10, marker='*')
    ax.text(t[-1] + 0.2, x[-1], '$\\mathcal{L}$', fontsize=9, color=ATLAS_COLORS['arrow_bwd'])
    
    # Backward arrows
    for idx in [95, 65, 35]:
        draw_flow_arrow(ax, t, x, idx, 'backward', ATLAS_COLORS['arrow_bwd'])
    
    ax.set_title('BACKWARD', fontsize=10, fontweight='bold', color=ATLAS_COLORS['arrow_bwd'], pad=3)
    ax.text(0.5, 0.02, '$T \\to 0$', transform=ax.transAxes, ha='center', fontsize=9,
            color=ATLAS_COLORS['arrow_bwd'])
    clean_axis(ax)


# =============================================================================
# Row 3: SDE vs ODE
# =============================================================================

def plot_sde(ax):
    """SDE with external noise."""
    t, x = generate_sde_trajectory(seed=42, n_steps=120, noise_scale=0.4)
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.8)
    
    # Noise kicks
    np.random.seed(42)
    noise_pts = np.random.choice(range(15, 105), size=8, replace=False)
    for pt in noise_pts:
        noise_mag = np.random.randn() * 0.25
        ax.annotate('', xy=(t[pt], x[pt] + noise_mag),
                    xytext=(t[pt], x[pt]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['noise'], lw=1.2, alpha=0.6))
    
    ax.set_title('SDE', fontsize=10, fontweight='bold', color=ATLAS_COLORS['noise'], pad=3)
    ax.text(0.5, 0.02, '$dW_t$ noise', transform=ax.transAxes, ha='center', fontsize=8,
            color=ATLAS_COLORS['noise'])
    clean_axis(ax)


def plot_ode(ax):
    """ODE with deterministic thermostat."""
    t, x, xi = generate_ode_trajectory(seed=42, n_steps=120)
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=1.8, alpha=0.8)
    ax.plot(t, xi * 0.4, color=ATLAS_COLORS['thermostat'], lw=1.2, alpha=0.5, ls='--')
    
    # Coupling indicators
    for pt in [30, 60, 90]:
        ax.plot([t[pt], t[pt]], [x[pt], xi[pt] * 0.4], 
               color=ATLAS_COLORS['thermostat'], ls=':', lw=1, alpha=0.5)
    
    ax.set_title('ODE', fontsize=10, fontweight='bold', color=ATLAS_COLORS['thermostat'], pad=3)
    ax.text(0.5, 0.02, '$\\xi$ thermostat', transform=ax.transAxes, ha='center', fontsize=8,
            color=ATLAS_COLORS['thermostat'])
    clean_axis(ax)


# =============================================================================
# Row 4: REINFORCE vs GIRSANOV
# =============================================================================

def plot_reinforce(ax):
    """REINFORCE - score function estimator."""
    n_paths = 12
    for i in range(n_paths):
        t, x = generate_sde_trajectory(seed=200 + i, n_steps=120, noise_scale=0.35)
        reward = np.exp(-0.5 * (x[-1] - 0.8)**2)
        color = plt.cm.RdYlGn(0.2 + 0.6 * reward)
        ax.plot(t, x, color=color, alpha=0.4, lw=1.2)
        ax.scatter(t[-1], x[-1], s=20 + 50 * reward, color=color, edgecolor='black', lw=0.3, zorder=5)
    
    # Target region
    ax.axhspan(0.3, 1.3, alpha=0.08, color=ATLAS_COLORS['equilibrium'])
    
    ax.set_title('REINFORCE', fontsize=10, fontweight='bold', color=ATLAS_COLORS['reinforce'], pad=3)
    ax.text(0.5, 0.02, '$R \\cdot \\nabla \\log p$', transform=ax.transAxes, ha='center', fontsize=8,
            color=ATLAS_COLORS['reinforce'])
    clean_axis(ax)


def plot_girsanov(ax):
    """Girsanov - path reweighting with M_T illustration."""
    t, x_ref = generate_sde_trajectory(seed=42, n_steps=120, drift_scale=0.05)
    ax.plot(t, x_ref, color=ATLAS_COLORS['gray'], lw=2.5, alpha=0.35, label='$\\mathbb{Q}$')
    
    # Perturbed path under P_θ
    np.random.seed(42)
    x_pert = np.zeros(120)
    x_pert[0] = x_ref[0]
    dt = t[1] - t[0]
    for i in range(1, 120):
        x_pert[i] = x_pert[i-1] - 0.15 * x_pert[i-1] * dt + 0.3 * np.sqrt(dt) * np.random.randn()
    ax.plot(t, x_pert, color=ATLAS_COLORS['girsanov'], lw=1.8, alpha=0.9, label='$\\mathbb{P}_\\theta$')
    
    # Compute and show M_t accumulating (as growing markers)
    # M_t = exp(∫ (b_θ - b_Q)/σ dW - 1/2 ∫ ((b_θ - b_Q)/σ)² dt)
    weight_pts = [0, 30, 60, 90, 119]
    M_values = [1.0]  # M_0 = 1
    for i in range(1, len(weight_pts)):
        # Simplified: M grows with drift difference accumulation
        drift_diff_accum = np.sum(np.abs(x_pert[:weight_pts[i]] - x_ref[:weight_pts[i]])) * 0.02
        M_values.append(np.exp(drift_diff_accum * 0.3))
    
    # Normalize for visualization
    M_values = np.array(M_values)
    M_normalized = (M_values - M_values.min()) / (M_values.max() - M_values.min())
    
    # Draw M_t as growing circles along the perturbed path
    for i, pt in enumerate(weight_pts):
        size = 15 + 60 * M_normalized[i]
        ax.scatter(t[pt], x_pert[pt], s=size, color=ATLAS_COLORS['girsanov'], 
                  edgecolor='white', linewidth=0.8, zorder=10, alpha=0.8)
    
    # Draw bracket showing M_T at the end
    ax.annotate('', xy=(t[-1] + 0.3, x_pert[-1] + 0.15),
                xytext=(t[-1] + 0.3, x_pert[-1] - 0.15),
                arrowprops=dict(arrowstyle=']-[', color=ATLAS_COLORS['girsanov'], lw=1.5))
    ax.text(t[-1] + 0.5, x_pert[-1], '$M_T$', fontsize=9, color=ATLAS_COLORS['girsanov'],
           fontweight='bold', va='center')
    
    ax.set_title('GIRSANOV', fontsize=10, fontweight='bold', color=ATLAS_COLORS['girsanov'], pad=3)
    ax.text(0.5, 0.02, '$M_T = \\frac{d\\mathbb{P}_\\theta}{d\\mathbb{Q}}$', 
            transform=ax.transAxes, ha='center', fontsize=8, color=ATLAS_COLORS['girsanov'])
    ax.set_xlim(t[0] - 0.2, t[-1] + 1.0)
    clean_axis(ax)


# =============================================================================
# Row 5: FINE vs COARSE (ML ↔ Observable Matching)
# =============================================================================

def plot_fine(ax):
    """Fine-grained: Maximum Likelihood - match individual samples."""
    # Generate model samples
    n_model = 15
    np.random.seed(300)
    
    # Data points (target)
    data_x = np.array([0.3, 0.8, 1.2, 1.6, 2.1, 2.5, 3.0, 3.4, 3.9, 4.3])
    data_y = np.sin(data_x * 0.8) * 0.8 + np.random.randn(len(data_x)) * 0.15
    
    # Model samples (from simulation)
    for i in range(n_model):
        t, x = generate_sde_trajectory(seed=300 + i, n_steps=80, noise_scale=0.3)
        # Fade based on distance to show "matching"
        ax.plot(t * 0.8, x * 0.6, color=ATLAS_COLORS['fine'], alpha=0.2, lw=0.8)
    
    # Data points highlighted
    ax.scatter(data_x * 0.8, data_y, s=50, color=ATLAS_COLORS['data'], 
              edgecolor='black', linewidth=0.8, zorder=10, marker='o')
    
    # Arrows from model to data (showing per-sample matching)
    for i in range(0, len(data_x), 2):
        ax.annotate('', xy=(data_x[i] * 0.8, data_y[i]),
                   xytext=(data_x[i] * 0.8, data_y[i] - 0.4),
                   arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['fine'], 
                                  lw=1, alpha=0.6))
    
    ax.set_title('FINE (ML)', fontsize=10, fontweight='bold', color=ATLAS_COLORS['fine'], pad=3)
    ax.text(0.5, 0.02, '$-\\log p_\\theta(x^{\\mathrm{data}})$', 
            transform=ax.transAxes, ha='center', fontsize=8, color=ATLAS_COLORS['fine'])
    clean_axis(ax)


def plot_coarse(ax):
    """Coarse-grained: Observable matching - match statistics only."""
    # Generate ensemble
    n_paths = 18
    all_endpoints = []
    
    for i in range(n_paths):
        t, x = generate_sde_trajectory(seed=400 + i, n_steps=100, noise_scale=0.32)
        weight = np.exp(-0.2 * (x[-1])**2)
        color = plt.cm.Blues(0.3 + 0.5 * weight)
        ax.plot(t, x, color=color, alpha=0.25, lw=0.9)
        all_endpoints.append(x[-1])
    
    # Show the observable: mean (coarse statistic)
    mean_endpoint = np.mean(all_endpoints)
    target_mean = 0.5
    
    # Horizontal lines showing ⟨O⟩_sim vs O_target
    ax.axhline(mean_endpoint, color=ATLAS_COLORS['coarse'], lw=2, ls='-', alpha=0.8)
    ax.axhline(target_mean, color=ATLAS_COLORS['data'], lw=2, ls='--', alpha=0.8)
    
    # Arrow showing the gap to minimize
    mid_t = t[-1] * 0.7
    ax.annotate('', xy=(mid_t, target_mean), xytext=(mid_t, mean_endpoint),
               arrowprops=dict(arrowstyle='<->', color=ATLAS_COLORS['annotation'], lw=1.5))
    
    # Labels
    ax.text(t[-1] + 0.2, mean_endpoint, '$\\langle O \\rangle$', fontsize=8, 
           color=ATLAS_COLORS['coarse'], va='center', fontweight='bold')
    ax.text(t[-1] + 0.2, target_mean, '$O^*$', fontsize=8, 
           color=ATLAS_COLORS['data'], va='center', fontweight='bold')
    
    ax.set_title('COARSE (Obs)', fontsize=10, fontweight='bold', color=ATLAS_COLORS['coarse'], pad=3)
    ax.text(0.5, 0.02, '$(\\langle O \\rangle - O^*)^2$', 
            transform=ax.transAxes, ha='center', fontsize=8, color=ATLAS_COLORS['coarse'])
    ax.set_xlim(t[0] - 0.3, t[-1] + 1.0)
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
        ('IV', 'Score vs Reweight'),
        ('V', 'Fine vs Coarse'),
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
    
    ax_reinforce = fig.add_subplot(gs[0, 3])
    ax_girsanov = fig.add_subplot(gs[1, 3])
    plot_reinforce(ax_reinforce)
    plot_girsanov(ax_girsanov)
    
    ax_fine = fig.add_subplot(gs[0, 4])
    ax_coarse = fig.add_subplot(gs[1, 4])
    plot_fine(ax_fine)
    plot_coarse(ax_coarse)
    
    # Column labels on top
    for i, (num, label) in enumerate(col_info):
        x_pos = 0.11 + i * 0.19
        fig.text(x_pos, 0.88, f'{num}. {label}', fontsize=8, fontweight='bold',
                color=ATLAS_COLORS['annotation'], ha='center')
    
    # "vs" labels between rows (in the middle of each column)
    for i in range(5):
        x_pos = 0.11 + i * 0.19
        fig.text(x_pos, 0.47, 'vs', fontsize=8, color=ATLAS_COLORS['gray'],
                ha='center', va='center', style='italic')
    
    # Main title
    fig.suptitle('The Gradient Method Atlas', fontsize=13, fontweight='bold',
                y=0.97, color=ATLAS_COLORS['annotation'])
    
    # Footer
    fig.text(0.5, 0.01,
             'All methods → $-\\beta\\,\\mathrm{Cov}_\\pi(\\mathcal{O}, \\nabla_\\theta U)$ at equilibrium',
             ha='center', fontsize=9, color=ATLAS_COLORS['equilibrium'], fontweight='bold')
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gradient_atlas.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gradient Atlas saved to: {output_path}")


if __name__ == "__main__":
    main()
