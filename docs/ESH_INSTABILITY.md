# ESH Sampler Numerical Instability

## Summary

The ESH (Energy Sampling Hamiltonian) integrator in `uni_diffsim` has a **numerical instability** that causes the log-velocity magnitude `r` to diverge to -∞ for certain trajectories.

## Root Cause

The instability occurs when:
1. **`u·(-e) ≈ -1`**: velocity points directly opposite to gradient (uphill)
2. **Large gradient**: `eps × |grad| / d >> 1`

In this regime, the term `Z = 1 + u·(-e) - A2` becomes very small, and `log(Z)` produces large negative spikes in `delta_r`.

## Symptoms

- `r` drifts to extreme negative values (-10^14)
- Importance weights `exp(r)` collapse to 0
- Effective Sample Size (ESS) becomes tiny (~1%)
- Resampled points cluster in early trajectory segments

## When It Happens

- Double-well potential: particles escape to high-gradient regions
- Any unbounded potential with steep gradients
- Long trajectories (accumulates over time)

## Workarounds

1. **Use smaller step size**: `eps × max(|grad|) / d < 1`
2. **Use gradient clipping**: `ESH(eps=0.1, max_grad_norm=10.0)`
3. **Use bounded potentials**: confining walls prevent escape
4. **Use shorter trajectories**: stop before divergence accumulates

## Status

The implementation is mathematically correct but numerically fragile. For production use, consider BAOAB or Nosé-Hoover which are more robust.

