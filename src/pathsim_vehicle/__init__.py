"""
PathSim-Vehicle: Vehicle Dynamics Blocks for PathSim

A toolbox providing specialized blocks for vehicle dynamics simulations
in the PathSim framework.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .parameters import BicycleParameters, DynamicBicycleParameters, hyundai_azera
from .constraints import BicycleConstraints, clip_steering, clip_acceleration
from .kinematic_bicycle import KinematicBicycle
from .dynamic_bicycle import DynamicBicycle

__all__ = [
    "__version__",
    "BicycleParameters",
    "DynamicBicycleParameters",
    "hyundai_azera",
    "BicycleConstraints",
    "clip_steering",
    "clip_acceleration",
    "KinematicBicycle",
    "DynamicBicycle",
]
