"""Kinematic bicycle model block for PathSim.

Implements Eq. (1) from: Kong, J., Pfeiffer, M., Schildbach, G., and Borrelli, F.
"Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design",
IEEE IV, 2015.

State vector: [x, y, psi, v]
    x   — inertial X position [m]
    y   — inertial Y position [m]
    psi — heading angle [rad]
    v   — speed [m/s]

Inputs: [delta_f, a]
    delta_f — front steering angle [rad]
    a       — acceleration [m/s²]
"""

from __future__ import annotations

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

from .parameters import BicycleParameters


class KinematicBicycle(DynamicalSystem):
    """Kinematic bicycle model (Eq. 1 in Kong et al. IV 2015).

    A simple nonlinear vehicle model that captures position, heading,
    and speed dynamics using only the wheelbase geometry. No tire model
    is required, making it valid across all speeds including zero.

    .. math::

        \\dot{x} = v \\cos(\\psi + \\beta)

        \\dot{y} = v \\sin(\\psi + \\beta)

        \\dot{\\psi} = \\frac{v}{l_r} \\sin(\\beta)

        \\dot{v} = a

    where :math:`\\beta = \\arctan\\!\\left(\\frac{l_r}{l_f + l_r} \\tan(\\delta_f)\\right)`.

    Parameters
    ----------
    params : BicycleParameters
        Vehicle geometry (l_f, l_r).
    x0 : float
        Initial X position [m].
    y0 : float
        Initial Y position [m].
    psi0 : float
        Initial heading [rad].
    v0 : float
        Initial speed [m/s].
    """

    input_port_labels = {
        "delta_f": 0,
        "a": 1,
    }

    output_port_labels = {
        "x": 0,
        "y": 1,
        "psi": 2,
        "v": 3,
    }

    def __init__(
        self,
        params: BicycleParameters,
        x0: float = 0.0,
        y0: float = 0.0,
        psi0: float = 0.0,
        v0: float = 0.0,
    ):
        self.params = params

        def _fn_dyn(state, u, t):
            x, y, psi, v = state
            delta_f, a = u

            l_f = self.params.l_f
            l_r = self.params.l_r

            # sideslip angle at center of gravity
            beta = np.arctan2(l_r * np.tan(delta_f), l_f + l_r)

            dx = v * np.cos(psi + beta)
            dy = v * np.sin(psi + beta)
            dpsi = (v / l_r) * np.sin(beta)
            dv = a

            return np.array([dx, dy, dpsi, dv])

        def _fn_alg(state, u, t):
            return state.copy()

        super().__init__(
            func_dyn=_fn_dyn,
            func_alg=_fn_alg,
            initial_value=np.array([x0, y0, psi0, v0]),
        )
