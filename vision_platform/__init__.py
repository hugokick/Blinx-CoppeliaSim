"""Shared visual-robot teaching platform for simulation and real hardware."""

from .config import VisionLabConfig, load_config
from .models import Detection, Frame, TaskEvent, TaskResult

__all__ = [
    "Detection",
    "Frame",
    "TaskEvent",
    "TaskResult",
    "VisionLabConfig",
    "load_config",
]
