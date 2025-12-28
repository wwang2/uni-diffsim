"""Unified differentiable molecular simulations."""

from .device import get_device, to_device, available_devices
from .potentials import Potential, DoubleWell, MullerBrown, LennardJones, Harmonic
from .integrators import (
    OverdampedLangevin, BAOAB, VelocityVerlet, NoseHooverChain, ESH, GLE,
    kinetic_energy, temperature,
)

__version__ = "0.1.0"
__all__ = [
    # Device
    "get_device", "to_device", "available_devices",
    # Potentials
    "Potential", "DoubleWell", "MullerBrown", "LennardJones", "Harmonic",
    # Integrators
    "OverdampedLangevin", "BAOAB", "VelocityVerlet", "NoseHooverChain", "ESH", "GLE",
    "kinetic_energy", "temperature",
]
