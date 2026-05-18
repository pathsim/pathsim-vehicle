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

        \\ddot{y} = -\\dot{\\psi} \\, \\dot{x} + \\frac{1}{m}(F_{c,f} \\cos \\delta_f + F_{c,r})

        \\ddot{\\psi} = \\frac{1}{I_z}(l_f F_{c,f} - l_r F_{c,r})

        \\dot{X} = \\dot{x} \\cos \\psi - \\dot{y} \\sin \\psi

        \\dot{Y} = \\dot{x} \\sin \\psi + \\dot{y} \\cos \\psi

    where the linear tire forces are :math:`F_{c,i} = -C_{\\alpha i} \\, \\alpha_i`
    and :math:`C_{\\alpha i}` is the total axle cornering stiffness (both tires combined).

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
            C_f = self.params.C_af
            C_r = self.params.C_ar

            # Use vx for computing dynamics, with floor for low-speed safety
            vx_safe = max(abs(vx), 1.0)

            # Standard 2-DOF lateral-yaw state-space model
            # d/dt [vy, psi_dot] = A * [vy, psi_dot] + B * delta_f
            #
            # A = [-(C_f+C_r)/(m*vx),    -vx - (l_f*C_f - l_r*C_r)/(m*vx)]
            #     [-(l_f*C_f-l_r*C_r)/Iz, -(l_f²*C_f + l_r²*C_r)/(Iz*vx) ]
            #
            # B = [C_f/m,       ]
            #     [l_f*C_f/Iz   ]

            dvx = a_x
            dvy = (-(C_f + C_r) / (m * vx_safe)) * vy \
                + (-vx_safe - (l_f * C_f - l_r * C_r) / (m * vx_safe)) * psi_dot \
                + (C_f / m) * delta_f
            dpsi = psi_dot
            dpsi_dot = (-(l_f * C_f - l_r * C_r) / I_z) * vy \
                     + (-(l_f**2 * C_f + l_r**2 * C_r) / (I_z * vx_safe)) * psi_dot \
                     + (l_f * C_f / I_z) * delta_f

            # Inertial position (nonlinear kinematics)
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
