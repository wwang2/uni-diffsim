"""Device utilities for CPU/CUDA/MPS support."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        preference: "auto", "cpu", "cuda", or "mps"
    
    Returns:
        torch.device for computation
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    
    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(x, device: torch.device | str | None = None):
    """Move tensor or module to device."""
    if device is None:
        device = get_device()
    if isinstance(device, str):
        device = torch.device(device)
    return x.to(device)


def available_devices() -> list[str]:
    """Return list of available device names."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices
