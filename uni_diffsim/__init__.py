"""Unified differentiable molecular simulations."""

from .device import get_device, to_device, available_devices
from .potentials import Potential, DoubleWell, AsymmetricDoubleWell, MullerBrown, LennardJones, Harmonic
from .integrators import (
    OverdampedLangevin, BAOAB, VelocityVerlet, NoseHooverChain, ESH, GLE,
    kinetic_energy, temperature,
)
from .gradient_estimators import (
    ReinforceEstimator, GirsanovEstimator, ReweightingLoss,
    ImplicitDiffEstimator,
    reinforce_gradient,
    CheckpointManager, CheckpointedNoseHoover, ContinuousAdjointNoseHoover,
)

__version__ = "0.1.0"
__all__ = [
    # Device
    "get_device", "to_device", "available_devices",
    # Potentials
    "Potential", "DoubleWell", "AsymmetricDoubleWell", "MullerBrown", "LennardJones", "Harmonic",
    # Integrators
    "OverdampedLangevin", "BAOAB", "VelocityVerlet", "NoseHooverChain", "ESH", "GLE",
    "kinetic_energy", "temperature",
    # Gradient Estimators
    "ReinforceEstimator", "GirsanovEstimator", "ReweightingLoss",
    "ImplicitDiffEstimator",
    "reinforce_gradient",
    # Adjoint Methods
    "CheckpointManager", "CheckpointedNoseHoover", "ContinuousAdjointNoseHoover",
]
