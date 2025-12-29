# Integrator Benchmarks

**Date:** 2025-12-28 11:46:24
**Device:** cpu
**Particles:** 1000, **Dimensions:** 100, **Steps:** 1000

| Integrator | Time (s) | Steps/sec | Peak Mem (MB) |
| :--- | :--- | :--- | :--- |
| OverdampedLangevin | 1.4386 | 695143 | 0.23 |
| BAOAB | 2.5173 | 397257 | 0.53 |
| VelocityVerlet | 0.5383 | 1857858 | 0.15 |
| NoseHoover | 1.5406 | 649110 | 0.84 |
| NoseHooverChain | 2.3681 | 422271 | 1.98 |
| ESH | 2.8744 | 347898 | 3.89 |
| GLE | 3.0065 | 332618 | 0.92 |
