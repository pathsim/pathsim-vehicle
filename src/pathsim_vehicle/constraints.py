"""Input constraints for bicycle vehicle models.

Based on Table II in: Kong, J., Pfeiffer, M., Schildbach, G., and Borrelli, F.
"Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design",
IEEE IV, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BicycleConstraints:
    """Input and rate constraints for the bicycle model (Table II in Kong et al.).

    All angular values are stored in radians.

    Parameters
    ----------
    delta_min : float
        Minimum steering angle [rad].
    delta_max : float
        Maximum steering angle [rad].
    a_min : float
        Minimum (braking) acceleration [m/s²].
    a_max : float
        Maximum acceleration [m/s²].
    delta_dot_min : float
        Minimum steering rate [rad/s].
    delta_dot_max : float
        Maximum steering rate [rad/s].
    a_dot_min : float
        Minimum jerk [m/s³].
    a_dot_max : float
        Maximum jerk [m/s³].
    """

    delta_min: float = -0.6458       # -37° in rad
    delta_max: float = 0.6458        # +37° in rad
    a_min: float = -1.5              # m/s²
    a_max: float = 1.0               # m/s²
    delta_dot_min: float = -0.17453  # -10°/s in rad/s
    delta_dot_max: float = 0.17453   # +10°/s in rad/s
    a_dot_min: float = -3.0          # m/s³
    a_dot_max: float = 1.5           # m/s³


def clip_steering(
    delta: float | np.ndarray,
    constraints: BicycleConstraints,
) -> float | np.ndarray:
    """Clip steering angle to constraint bounds.

    Parameters
    ----------
    delta : float or ndarray
        Steering angle [rad].
    constraints : BicycleConstraints
        Constraint bounds.

    Returns
    -------
    float or ndarray
        Clipped steering angle.
    """
    return np.clip(delta, constraints.delta_min, constraints.delta_max)


def clip_acceleration(
    a: float | np.ndarray,
    constraints: BicycleConstraints,
) -> float | np.ndarray:
    """Clip acceleration to constraint bounds.

    Parameters
    ----------
    a : float or ndarray
        Acceleration [m/s²].
    constraints : BicycleConstraints
        Constraint bounds.

    Returns
    -------
    float or ndarray
        Clipped acceleration.
    """
    return np.clip(a, constraints.a_min, constraints.a_max)
