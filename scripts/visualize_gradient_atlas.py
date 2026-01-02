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
    """Single path - remembers everything. Pathwise/reparameterization gradient.
    
    Pathwise: E[∇_x O · ∂x_T/∂θ] - explicit Jacobian Φ(T,t)
    Variance ~ exp(2λT) due to Lyapunov instability
    """
    t, x = generate_sde_trajectory(seed=42, n_steps=120)
    ax.plot(t, x, color=ATLAS_COLORS['path_single'], lw=2, alpha=0.9)
    
    # Mark sensitivity chain - Jacobian Φ(T,t) propagates through history
    events = [20, 50, 80, 100]
    for ev in events:
        ax.scatter(t[ev], x[ev], s=40, color=ATLAS_COLORS['arrow_bwd'], zorder=10)
    for i in range(len(events) - 1):
        ax.annotate('', xy=(t[events[i+1]], x[events[i+1]]),
                    xytext=(t[events[i]], x[events[i]]),
                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['sensitivity'],
                                   lw=1.2, ls='--', connectionstyle='arc3,rad=0.15'))
    
    ax.set_title('PATH', fontsize=11, fontweight='bold', color=ATLAS_COLORS['path_single'], pad=4)
    # Show the pathwise formula
    ax.text(0.5, 0.02, '"remembers"', transform=ax.transAxes, ha='center', fontsize=9,
            style='italic', color=ATLAS_COLORS['annotation'])
    clean_axis(ax)


def plot_ensemble(ax):
    """Ensemble - forgets history. Score function/REINFORCE gradient.
    
    REINFORCE: E[O · ∇_θ log p(τ)] - score function estimator
    Variance ~ T (Itô isometry) - much better than exp(2λT)!
    
    KEY: Malliavin IBP shows PATH ↔ ENSEMBLE are equivalent:
    E[∇_x O · ∂x_T/∂θ] = E[O · ∇_θ log p(τ)]
    The Jacobian is "hidden" in the score!
    """
    n_paths = 20
    for i in range(n_paths):
        t, x = generate_sde_trajectory(seed=100 + i, n_steps=120, noise_scale=0.35)
        weight = np.exp(-0.3 * x[-1]**2)
        color = plt.cm.Blues(0.3 + 0.5 * weight)
        ax.plot(t, x, color=color, alpha=0.3 + 0.4 * weight, lw=1)
    
    # Equilibrium distribution - the ONLY thing that matters for ensemble view
    y_range = np.linspace(-2, 2, 80)
    eq_dist = np.exp(-0.5 * y_range**2)
    eq_dist = eq_dist / eq_dist.max() * 0.8
    ax.fill_betweenx(y_range, t[-1], t[-1] + eq_dist, color=ATLAS_COLORS['equilibrium'], alpha=0.4)
    ax.plot(t[-1] + eq_dist, y_range, color=ATLAS_COLORS['equilibrium'], lw=1.5)
    ax.text(t[-1] + 0.5, 0, '$\\pi$', fontsize=10, color=ATLAS_COLORS['equilibrium'], fontweight='bold')
    
    ax.set_title('ENSEMBLE', fontsize=11, fontweight='bold', color=ATLAS_COLORS['ensemble_base'], pad=4)
    # Emphasize this is space average over distribution (history forgotten)
    ax.text(0.5, 0.02, '"forgets"', transform=ax.transAxes, ha='center', fontsize=9,
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
    
    ax.set_title('FORWARD', fontsize=11, fontweight='bold', color=ATLAS_COLORS['arrow_fwd'], pad=4)
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
    
    ax.set_title('BACKWARD', fontsize=11, fontweight='bold', color=ATLAS_COLORS['arrow_bwd'], pad=4)
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
    
    ax.set_title('SDE', fontsize=11, fontweight='bold', color=ATLAS_COLORS['noise'], pad=4)
    ax.text(0.5, 0.02, '$dW_t$ noise', transform=ax.transAxes, ha='center', fontsize=9,
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
    
    ax.set_title('ODE', fontsize=11, fontweight='bold', color=ATLAS_COLORS['thermostat'], pad=4)
    ax.text(0.5, 0.02, '$\\xi$ thermostat', transform=ax.transAxes, ha='center', fontsize=9,
            color=ATLAS_COLORS['thermostat'])
    clean_axis(ax)


# =============================================================================
# Row 4: REINFORCE vs GIRSANOV
# =============================================================================

def plot_reinforce(ax):
    """REINFORCE - Trajectories → Final samples → O · ∇_θU → Gradient.
    
    Key insight: For equilibrium observables, the gradient is:
        ∇_θ ⟨O⟩ = -β Cov(O, ∇_θU) = -β ⟨(O - ⟨O⟩) · ∇_θU⟩
    
    Trajectory endpoints become samples. Each sample contributes:
        O(xᵢ) · (-β ∇_θU(xᵢ))
    """
    np.random.seed(200)
    
    # =========================================================================
    # Background: Show the observable O(x) as a hint
    # =========================================================================
    y_range_full = np.linspace(-1.8, 1.8, 100)
    # Observable O(x) = x (simple linear observable for schematic)
    # Let's show it as a subtle gradient or color bar
    ax.fill_between([0, 2.5], -1.8, 1.8, color=ATLAS_COLORS['fill'], alpha=0.15, zorder=0)
    
    # =========================================================================
    # Generate trajectories and collect endpoints as samples
    # =========================================================================
    n_traj = 6
    n_steps = 60
    t_max = 1.6
    
    endpoints = []
    for i in range(n_traj):
        t, x = generate_sde_trajectory(seed=200 + i, n_steps=n_steps, noise_scale=0.35)
        t_scaled = t / t.max() * t_max
        ax.plot(t_scaled, x, color=ATLAS_COLORS['reinforce'], alpha=0.18, lw=0.8)
        endpoints.append(x[-1])
    
    endpoints = np.array(endpoints)
    x_sample = t_max + 0.1
    
    # =========================================================================
    # Mark trajectory endpoints and their contributions
    # =========================================================================
    for i, y in enumerate(endpoints):
        # Sample point
        ax.scatter(x_sample, y, s=50, color=ATLAS_COLORS['reinforce'], 
                  edgecolor='white', linewidth=1.2, zorder=10)
        
        # O · ∇_θU arrow (the "score contribution")
        # For x^2 observable and Gaussian p, score is proportional to x
        arrow_len = y * 0.25
        if abs(arrow_len) > 0.05:
            ax.annotate('', xy=(x_sample + arrow_len, y), xytext=(x_sample, y),
                       arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['reinforce'], 
                                      lw=1.8, alpha=0.9, mutation_scale=9))
    
    # Label for samples
    ax.text(x_sample, 1.6, '$x_i$', fontsize=8, ha='center',
            color=ATLAS_COLORS['reinforce'], fontweight='bold')
    
    # =========================================================================
    # RIGHT: Funneling to the aggregate gradient
    # =========================================================================
    x_funnel = x_sample + 0.5
    x_grad = x_funnel + 0.35
    
    # Funnel lines
    for y in endpoints:
        arrow_len = y * 0.25
        ax.plot([x_sample + arrow_len, x_funnel], [y, 0], 
               color=ATLAS_COLORS['reinforce'], lw=0.6, alpha=0.15, ls='--')
    
    # Gradient aggregate arrow
    ax.annotate('', xy=(x_grad + 0.3, 0), xytext=(x_funnel, 0),
                arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['reinforce'], 
                               lw=2.8, mutation_scale=12))
    
    # Labels
    # ax.text(x_grad + 0.15, 0.25, '$\\nabla_\\theta\\langle O\\rangle$', fontsize=8,
    #         color=ATLAS_COLORS['reinforce'], fontweight='bold', ha='center')
    
    ax.text(x_grad - 0.1, -0.6, '$-\\beta\\mathrm{Cov}(O, \\nabla_\\theta U)$', 
            fontsize=7, ha='center', color=ATLAS_COLORS['reinforce'], style='italic')

    ax.set_title('REINFORCE (Score)', fontsize=11, fontweight='bold', 
                 color=ATLAS_COLORS['reinforce'], pad=4)
    
    ax.set_xlim(-0.1, x_grad + 0.6)
    ax.set_ylim(-1.8, 1.8)
    clean_axis(ax)


def plot_girsanov(ax):
    """Girsanov - Ensemble of paths with LOCAL accumulation of log p(τ).
    
    Key insight: For path-dependent observables O(τ), the gradient is:
        ∇_θ ⟨O⟩ = ⟨O(τ) · ∇_θ log p(τ|θ)⟩
    
    where the path score is a PATH INTEGRAL that accumulates locally:
        ∇_θ log p(τ) = (1/σ²) Σₜ ∇_θb(xₜ) · ΔWₜ
    
    Path thickness reflects accumulated weight. 
    """
    np.random.seed(42)
    n_steps = 70
    dt = 0.04
    t = np.linspace(0, n_steps * dt, n_steps)
    
    # =========================================================================
    # Generate ENSEMBLE of paths with tracked weight accumulation
    # =========================================================================
    n_paths = 8
    paths = []
    path_weights = []
    
    for p in range(n_paths):
        x = np.zeros(n_steps)
        log_weight = np.zeros(n_steps)
        np.random.seed(300 + p)
        cumulative = 0.0
        for j in range(1, n_steps):
            dW = np.random.randn()
            drift = -0.15 * x[j-1]
            x[j] = x[j-1] + drift * dt + 0.35 * np.sqrt(dt) * dW
            
            # Local accumulation: Δ log p ∝ drift_gradient · dW
            # Using a simplified model where drift gradient is constant
            local_contrib = 0.25 * dW 
            cumulative += local_contrib
            log_weight[j] = cumulative
        
        paths.append(x)
        path_weights.append(log_weight)
    
    # =========================================================================
    # Draw paths with THICKNESS proportional to accumulated weight
    # =========================================================================
    all_final_weights = [pw[-1] for pw in path_weights]
    w_min, w_max = min(all_final_weights), max(all_final_weights)
    w_range = w_max - w_min + 0.01
    
    for p, (x, log_w) in enumerate(zip(paths, path_weights)):
        # Very distinct linewidth variation
        w_norm = (log_w[-1] - w_min) / w_range
        lw = 0.5 + 3.5 * (w_norm**1.5)  # non-linear for more contrast
        alpha = 0.15 + 0.65 * w_norm
        ax.plot(t, x, color=ATLAS_COLORS['girsanov'], lw=lw, alpha=alpha)
        
        # Observable marker at end
        ax.scatter(t[-1], x[-1], s=25, color=ATLAS_COLORS['data'], 
                  edgecolor='black', linewidth=0.8, zorder=15)

    # =========================================================================
    # Path Integral Detail: Show accumulation on ONE path
    # =========================================================================
    main_idx = np.argsort(all_final_weights)[-2] # pick a high-weight path
    main_path = paths[main_idx]
    
    checkpoints = [10, 22, 34, 46, 58]
    for cp in checkpoints:
        x_c, y_c = t[cp], main_path[cp]
        # "Score packet" - a small pulse showing local contribution
        pulse_x = np.linspace(-0.06, 0.06, 10)
        pulse_y = 0.12 * np.exp(-pulse_x**2 / 0.001)
        ax.plot(x_c + pulse_x, y_c + pulse_y, color="black", lw=1.5, alpha=0.8)
        
    # Labels for accumulation
    ax.text(t[checkpoints[1]], main_path[checkpoints[1]] + 0.3, '$\\delta \\ell_t$', 
            fontsize=8, color=ATLAS_COLORS['noise'], ha='center')
    
    # Label for path integral formula
    ax.text(t[-1], -1.3, '$\\langle O(\\tau) \\cdot \\int \\nabla_\\theta b \\cdot dW_t \\rangle$', 
            fontsize=7.5, color=ATLAS_COLORS['girsanov'], ha='left', style='italic')
    
    ax.set_title('GIRSANOV (Reweight)', fontsize=11, fontweight='bold', 
                 color=ATLAS_COLORS['girsanov'], pad=4)
    
    ax.set_xlim(-0.05, t[-1] + 0.8)
    ax.set_ylim(-1.8, 1.8)
    clean_axis(ax)


# =============================================================================
# Row 5: FINE vs COARSE (ML ↔ Observable Matching)
# =============================================================================

def plot_fine(ax):
    """Fine-grained: Maximum Likelihood - simulation guided toward data configurations.
    
    Shows trajectories being "pulled" toward data points (target configurations).
    The gradient is: E_data[∇U] - E_sim[∇U] (contrastive divergence).
    """
    np.random.seed(300)
    
    # Generate simulation trajectories
    n_traj = 12
    for i in range(n_traj):
        t, x = generate_sde_trajectory(seed=300 + i, n_steps=100, noise_scale=0.32)
        ax.plot(t, x, color=ATLAS_COLORS['fine'], alpha=0.2, lw=0.8)
    
    # Data points (target configurations) - shown at the END of trajectory space
    # These are the x^data we want to maximize likelihood of
    data_targets = np.array([-0.6, 0.1, 0.5, 0.9, 1.3])
    t_end = 5.0  # end of trajectory
    
    for y_data in data_targets:
        # Data point marker (yellow circle)
        ax.scatter(t_end , y_data, s=25, color=ATLAS_COLORS['data'], 
                  edgecolor='black', linewidth=1, zorder=10, marker='o')
    
    # Show model distribution p_θ at the end (where simulation lands)
    y_range = np.linspace(-2, 2, 80)
    # Current model distribution (slightly misaligned with data)
    model_dist = np.exp(-0.5 * (y_range - 0.2)**2 / 0.7**2)
    model_dist = model_dist / model_dist.max() * 0.6
    ax.fill_betweenx(y_range, t_end, t_end + model_dist, 
                     color=ATLAS_COLORS['fine'], alpha=0.3)
    ax.plot(t_end  + model_dist, y_range, color=ATLAS_COLORS['fine'], lw=1.2, alpha=0.7)
    
    # # Arrows from model distribution toward data points (showing gradient direction)
    # # This is the "contrastive" part: push probability toward data
    # for y_data in data_targets[1:4]:  # show a few arrows
    #     # Arrow from current model peak toward data
    #     model_y = 0.2  # current model mean
    #     if abs(y_data - model_y) > 0.2:
    #         ax.annotate('', xy=(t_end - 0.3, y_data),
    #                    xytext=(t_end - 0.3, model_y + 0.3 * np.sign(y_data - model_y)),
    #                    arrowprops=dict(arrowstyle='->', color=ATLAS_COLORS['fine'], 
    #                                   lw=1.3, alpha=0.7))
    
    # Labels
    ax.text(t_end + 0.8, 1.0, '$p_\\theta$', fontsize=8, color=ATLAS_COLORS['fine'], fontweight='bold')
    ax.text(t_end + 0.15, -1.5, '$x^{\\mathrm{data}}$', fontsize=8, color=ATLAS_COLORS['data'])
    
    ax.set_xlim(-0.3, t_end + 1.3)
    ax.set_ylim(-2, 2)
    ax.set_title('FINE (data likelihood)', fontsize=11, fontweight='bold', color=ATLAS_COLORS['fine'], pad=4)
    ax.text(0.5, 0.02, '$-\\log p_\\theta(x^{\\mathrm{data}})$', 
            transform=ax.transAxes, ha='center', fontsize=8, color=ATLAS_COLORS['fine'])
    clean_axis(ax)


def plot_coarse(ax):
    """Coarse-grained: Observable matching - match average position ⟨x⟩.
    
    Shows matching a COARSE observable like mean position ⟨x⟩ rather than 
    individual sample positions. Only care about the statistic, not the samples.
    """
    np.random.seed(400)
    
    # Generate ensemble
    n_paths = 15
    all_endpoints = []
    
    for i in range(n_paths):
        t, x = generate_sde_trajectory(seed=400 + i, n_steps=80, noise_scale=0.35)
        ax.plot(t, x, color=ATLAS_COLORS['coarse'], alpha=0.2, lw=0.8)
        all_endpoints.append(x[-1])
    
    all_endpoints = np.array(all_endpoints)
    
    # =========================================================================
    # Show the COARSE observable: average position ⟨x⟩
    # =========================================================================
    
    # Current model mean
    model_mean = np.mean(all_endpoints)
    
    # Target mean (what we want to match)
    target_mean = 0.6
    
    t_end = t[-1]
    
    # Horizontal lines showing ⟨x⟩_sim vs ⟨x⟩_target
    ax.axhline(model_mean, xmin=0.7, xmax=0.95, color=ATLAS_COLORS['coarse'], 
              lw=2.5, ls='-', alpha=0.9)
    ax.axhline(target_mean, xmin=0.7, xmax=0.95, color=ATLAS_COLORS['data'], 
              lw=2.5, ls='--', alpha=0.9)
    
    # Vertical arrow showing the gap to minimize
    arrow_x = t_end * 0.85
    ax.annotate('', xy=(arrow_x, target_mean), xytext=(arrow_x, model_mean),
               arrowprops=dict(arrowstyle='<->', color=ATLAS_COLORS['annotation'], 
                              lw=1.8, mutation_scale=10))
    
    # Labels
    ax.text(t_end + 0.15, model_mean, '$\\langle x \\rangle$', fontsize=8, 
           color=ATLAS_COLORS['coarse'], va='center', fontweight='bold')
    ax.text(t_end + 0.15, target_mean, '$x^*$', fontsize=8, 
           color=ATLAS_COLORS['data'], va='center', fontweight='bold')
    
    # Scatter endpoints to show the distribution (but we only care about mean!)
    for ep in all_endpoints:
        ax.scatter(t_end + 0.05, ep, s=18, color=ATLAS_COLORS['coarse'], 
                  edgecolor='white', linewidth=0.4, alpha=0.5, zorder=5)
    
    # Emphasize: we only care about the MEAN, not individual samples
    ax.text(t_end * 0.5, 1.45, 'only $\\langle x \\rangle$ matters', fontsize=6,
           color=ATLAS_COLORS['gray'], ha='center', style='italic', alpha=0.7)
    
    ax.set_title('COARSE (Obs)', fontsize=11, fontweight='bold', color=ATLAS_COLORS['coarse'], pad=4)
    ax.text(0.5, 0.02, '$(\\langle x \\rangle - x^*)^2$', 
            transform=ax.transAxes, ha='center', fontsize=8, color=ATLAS_COLORS['coarse'])
    ax.set_xlim(t[0] - 0.2, t_end + 0.7)
    ax.set_ylim(-1.8, 1.8)
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
    
    # Column labels (updated IV to clarify the distinction)
    col_info = [
        ('I', 'Path vs Ensemble'),
        ('II', 'Forward vs Backward'),
        ('III', 'SDE vs ODE'),
        ('IV', 'Eq. Score vs Path Reweight'),
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
    fig.text(x_ibp + 0.05, 0.47, 'Malliavin\n   IBP', fontsize=6, ha='left', va='center', 
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
             fontsize=6.5, ha='left', va='center', color=ATLAS_COLORS['arrow_bwd'], 
             style='italic')
    fig.text(x_ibp + 0.065, 0.395, 'var $\\sim T$', 
             fontsize=6.5, ha='left', va='center', color=ATLAS_COLORS['equilibrium'], 
             style='italic')
    
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
