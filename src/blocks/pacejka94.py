#########################################################################################
##
##                                  PACEJKA94 BLOCK
##                                (blocks/pacejka94.py)
##
#########################################################################################

import numpy as np
from pathsim.blocks import Block


PACEJKA94_LONGITUDINAL_DEFAULTS = {
    "b0": 1.5, "b1": 0.0, "b2": 1100.0, "b3": 0.0, "b4": 300.0, "b5": 0.0,
    "b6": 0.0, "b7": 0.0, "b8": -2.0, "b9": 0.0, "b10": 0.0, "b11": 0.0,
    "b12": 0.0, "b13": 0.0,
}

PACEJKA94_LATERAL_DEFAULTS = {
    "a0": 1.4, "a1": 0.0, "a2": 1100.0, "a3": 1100.0, "a4": 10.0, "a5": 0.0,
    "a6": 0.0, "a7": -2.0, "a8": 0.0, "a9": 0.0, "a10": 0.0, "a11": 0.0,
    "a12": 0.0, "a13": 0.0, "a14": 0.0, "a15": 0.0, "a16": 0.0, "a17": 0.0,
}


class Pacejka94(Block):
    """Compute tire longitudinal and lateral force using Pacejka '94 magic formula.

    Inputs
    ------
    slip_angle_deg : float
        Slip angle in degrees.
    slip_ratio_pct : float
        Slip ratio in percent.
    fz_N : float
        Vertical load in Newtons.
    camber_deg : float
        Camber angle in degrees. If input not given, set to zero.

    Outputs
    -------
    fx_N : float
        Longitudinal tire force in Newtons.
    fy_N : float
        Lateral tire force in Newtons.
    """

    input_port_labels = {"slip_angle_deg": 0, "slip_ratio_pct": 1, "fz_N": 2, "camber_deg": 3}
    output_port_labels = {"fx_N": 0, "fy_N": 1}

    def __init__(self, longitudinal_params=None, lateral_params=None, camber_deg=0.0):
        super().__init__()

        self.longitudinal_params = dict(PACEJKA94_LONGITUDINAL_DEFAULTS)
        self.lateral_params = dict(PACEJKA94_LATERAL_DEFAULTS)
        self.camber_deg = camber_deg

        if longitudinal_params is not None:
            if not isinstance(longitudinal_params, dict):
                raise ValueError("'longitudinal_params' must be dict or None")
            self.longitudinal_params.update(longitudinal_params)

        if lateral_params is not None:
            if not isinstance(lateral_params, dict):
                raise ValueError("'lateral_params' must be dict or None")
            self.lateral_params.update(lateral_params)

    def __len__(self):
        """Purely algebraic block."""
        return 1

    def _fx(self, slip_ratio_pct, fz_kN):
        p = self.longitudinal_params

        c = p["b0"]
        d = fz_kN * (p["b1"] * fz_kN + p["b2"])
        bcd = (p["b3"] * fz_kN**2 + p["b4"] * fz_kN) * np.exp(-p["b5"] * fz_kN)
        h = p["b9"] * fz_kN + p["b10"]
        v = p["b11"] * fz_kN + p["b12"]

        cd = c * d
        b = np.where(np.abs(cd) > 1e-12, bcd / cd, 0.0)
        bx1 = b * (slip_ratio_pct + h)
        e = (p["b6"] * fz_kN**2 + p["b7"] * fz_kN + p["b8"]) * (1.0 - p["b13"] * np.sign(slip_ratio_pct + h))

        return d * np.sin(c * np.arctan(bx1 - e * (bx1 - np.arctan(bx1)))) + v

    def _fy(self, slip_angle_deg, fz_kN, camber_deg):
        p = self.lateral_params

        c = p["a0"]
        d = fz_kN * (p["a1"] * fz_kN + p["a2"]) * (1.0 - p["a15"] * camber_deg**2)
        bcd = p["a3"] * np.sin(2.0 * np.arctan(fz_kN / p["a4"])) * (1.0 - p["a5"] * np.abs(camber_deg))
        h = p["a8"] * fz_kN + p["a9"] + p["a10"] * camber_deg
        v = p["a11"] * fz_kN + p["a12"] + (p["a13"] * fz_kN + p["a14"]) * camber_deg * fz_kN

        cd = c * d
        b = np.where(np.abs(cd) > 1e-12, bcd / cd, 0.0)
        bx1 = b * (slip_angle_deg + h)
        e = (p["a6"] * fz_kN + p["a7"]) * (1.0 - (p["a16"] * camber_deg + p["a17"]) * np.sign(slip_angle_deg + h))

        return d * np.sin(c * np.arctan(bx1 - e * (bx1 - np.arctan(bx1)))) + v

    def update(self, t):
        """Update algebraic output from current inputs."""
        u = self.inputs.to_array()

        slip_angle_deg = np.asarray(u[0], dtype=float)
        slip_ratio_pct = np.asarray(u[1], dtype=float)
        fz_kN = np.asarray(u[2], dtype=float) / 1000.0
        camber_deg = np.asarray(u[3], dtype=float) if len(u) > 3 else np.asarray(self.camber_deg, dtype=float)

        fx_N = self._fx(slip_ratio_pct, fz_kN)
        fy_N = self._fy(slip_angle_deg, fz_kN, camber_deg)
        self.outputs.update_from_array([fx_N, fy_N])
