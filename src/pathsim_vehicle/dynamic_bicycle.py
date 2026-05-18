"""Dynamic bicycle model block for PathSim.

Implements Eq. (2) from: Kong, J., Pfeiffer, M., Schildbach, G., and Borrelli, F.
"Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design",
IEEE IV, 2015.

State vector: [vx, vy, psi, psi_dot, X, Y]
    vx       — longitudinal velocity in body frame [m/s]
    vy       — lateral velocity in body frame [m/s]
    psi      — heading angle [rad]
    psi_dot  — yaw rate [rad/s]
    X        — inertial X position [m]
    Y        — inertial Y position [m]

Inputs: [delta_f, a_x]
    delta_f — front steering angle [rad]
    a_x     — longitudinal acceleration [m/s²]

Tire model: linear (Eq. 3), F_c,i = -C_alpha_i * alpha_i
"""

from __future__ import annotations

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

from .parameters import DynamicBicycleParameters


class DynamicBicycle(DynamicalSystem):
    """Dynamic bicycle model with linear tire forces (Eq. 2-3 in Kong et al. IV 2015).

    A higher-fidelity vehicle model that captures lateral tire slip effects.
    Uses body-frame velocities and a linear tire model for the lateral forces.

    .. math::

        \\ddot{x} = \\dot{\\psi} \\, \\dot{y} + a_x

        \\ddot{y} = -\\dot{\\psi} \\, \\dot{x} + \\frac{2}{m}(F_{c,f} \\cos \\delta_f + F_{c,r})

        \\ddot{\\psi} = \\frac{2}{I_z}(l_f F_{c,f} - l_r F_{c,r})

        \\dot{X} = \\dot{x} \\cos \\psi - \\dot{y} \\sin \\psi

        \\dot{Y} = \\dot{x} \\sin \\psi + \\dot{y} \\cos \\psi

    where the linear tire forces are :math:`F_{c,i} = -C_{\\alpha i} \\, \\alpha_i`.

    Parameters
    ----------
    params : DynamicBicycleParameters
        Vehicle parameters (l_f, l_r, m, I_z, C_af, C_ar).
    vx0 : float
        Initial longitudinal velocity [m/s].
    vy0 : float
        Initial lateral velocity [m/s].
    psi0 : float
        Initial heading angle [rad].
    psi_dot0 : float
        Initial yaw rate [rad/s].
    X0 : float
        Initial inertial X position [m].
    Y0 : float
        Initial inertial Y position [m].
    """

    input_port_labels = {
        "delta_f": 0,
        "a_x": 1,
    }

    output_port_labels = {
        "vx": 0,
        "vy": 1,
        "psi": 2,
        "psi_dot": 3,
        "X": 4,
        "Y": 5,
    }

    def __init__(
        self,
        params: DynamicBicycleParameters,
        vx0: float = 0.0,
        vy0: float = 0.0,
        psi0: float = 0.0,
        psi_dot0: float = 0.0,
        X0: float = 0.0,
        Y0: float = 0.0,
    ):
        self.params = params

        def _fn_dyn(state, u, t):
            vx, vy, psi, psi_dot, X, Y = state
            delta_f, a_x = u

            l_f = self.params.l_f
            l_r = self.params.l_r
            m = self.params.m
            I_z = self.params.I_z
            C_af = self.params.C_af
            C_ar = self.params.C_ar

            # Tire slip angles
            # Guard against zero longitudinal velocity
            if abs(vx) > 0.5:
                alpha_f = delta_f - np.arctan2(vy + l_f * psi_dot, vx)
                alpha_r = -np.arctan2(vy - l_r * psi_dot, vx)
            else:
                alpha_f = 0.0
                alpha_r = 0.0

            # Linear tire forces (Eq. 3)
            F_cf = -C_af * alpha_f
            F_cr = -C_ar * alpha_r

            # Equations of motion (Eq. 2)
            dvx = psi_dot * vy + a_x
            dvy = -psi_dot * vx + (2.0 / m) * (F_cf * np.cos(delta_f) + F_cr)
            dpsi = psi_dot
            dpsi_dot = (2.0 / I_z) * (l_f * F_cf - l_r * F_cr)
            dX = vx * np.cos(psi) - vy * np.sin(psi)
            dY = vx * np.sin(psi) + vy * np.cos(psi)

            return np.array([dvx, dvy, dpsi, dpsi_dot, dX, dY])

        def _fn_alg(state, u, t):
            return state.copy()

        super().__init__(
            func_dyn=_fn_dyn,
            func_alg=_fn_alg,
            initial_value=np.array([vx0, vy0, psi0, psi_dot0, X0, Y0]),
        )
