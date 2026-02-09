import os
import sys
import unittest

import numpy as np

# local module path (src/blocks is not a package in this repository)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "blocks"))

from pacejka94 import Pacejka94
from _embedding import Embedding


def _pacejka94_fx_ref(slip_ratio_pct, fz_kN):
    b0 = 1.5
    b1 = 0.0
    b2 = 1100.0
    b3 = 0.0
    b4 = 300.0
    b5 = 0.0
    b6 = 0.0
    b7 = 0.0
    b8 = -2.0
    b9 = 0.0
    b10 = 0.0
    b11 = 0.0
    b12 = 0.0
    b13 = 0.0

    c = b0
    d = fz_kN * (b1 * fz_kN + b2)
    bcd = (b3 * fz_kN**2 + b4 * fz_kN) * np.exp(-b5 * fz_kN)
    h = b9 * fz_kN + b10
    v = b11 * fz_kN + b12
    cd = c * d
    b = bcd / cd if abs(cd) > 1e-12 else 0.0
    bx1 = b * (slip_ratio_pct + h)
    e = (b6 * fz_kN**2 + b7 * fz_kN + b8) * (1.0 - b13 * np.sign(slip_ratio_pct + h))
    return d * np.sin(c * np.arctan(bx1 - e * (bx1 - np.arctan(bx1)))) + v


def _pacejka94_fy_ref(slip_angle_deg, fz_kN, camber_deg=0.0):
    a0 = 1.4
    a1 = 0.0
    a2 = 1100.0
    a3 = 1100.0
    a4 = 10.0
    a5 = 0.0
    a6 = 0.0
    a7 = -2.0
    a8 = 0.0
    a9 = 0.0
    a10 = 0.0
    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a14 = 0.0
    a15 = 0.0
    a16 = 0.0
    a17 = 0.0

    c = a0
    d = fz_kN * (a1 * fz_kN + a2) * (1.0 - a15 * camber_deg**2)
    bcd = a3 * np.sin(2.0 * np.arctan(fz_kN / a4)) * (1.0 - a5 * abs(camber_deg))
    h = a8 * fz_kN + a9 + a10 * camber_deg
    v = a11 * fz_kN + a12 + (a13 * fz_kN + a14) * camber_deg * fz_kN
    cd = c * d
    b = bcd / cd if abs(cd) > 1e-12 else 0.0
    bx1 = b * (slip_angle_deg + h)
    e = (a6 * fz_kN + a7) * (1.0 - (a16 * camber_deg + a17) * np.sign(slip_angle_deg + h))
    return d * np.sin(c * np.arctan(bx1 - e * (bx1 - np.arctan(bx1)))) + v


class TestPacejka94(unittest.TestCase):
    """Test the implementation of the 'Pacejka94' block class."""

    def test_embedding(self):
        p = Pacejka94()

        def src(t):
            return 2.0 * t - 10.0, 1.5 * t - 7.5, 4000.0 + 50.0 * t, 0.5 * np.sin(t)

        def ref(t):
            sa_deg, sr_pct, fz_N, camber_deg = src(t)
            fz_kN = fz_N / 1000.0
            fx = _pacejka94_fx_ref(sr_pct, fz_kN)
            fy = _pacejka94_fy_ref(sa_deg, fz_kN, camber_deg=camber_deg)
            return np.array([fx, fy], dtype=float)

        e = Embedding(p, src, ref)

        for t in range(1, 10):
            y, r = e.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(y - r), 0.0, places=8)

    def test_linearization(self):
        """test linearization and delinearization"""

        p = Pacejka94()

        def src(t):
            return 2.0 * t - 10.0, 1.5 * t - 7.5, 4200.0 + 20.0 * t, 0.25 * np.cos(t)

        def ref(t):
            sa_deg, sr_pct, fz_N, camber_deg = src(t)
            fz_kN = fz_N / 1000.0
            fx = _pacejka94_fx_ref(sr_pct, fz_kN)
            fy = _pacejka94_fy_ref(sa_deg, fz_kN, camber_deg=camber_deg)
            return np.array([fx, fy], dtype=float)

        e = Embedding(p, src, ref)

        for t in range(10):
            y, r = e.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(y - r), 0.0, places=8)

        p.linearize(3)

        for t in range(10):
            y, r = e.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(y - r), 0.0, places=8)

        p.delinearize()

        for t in range(10):
            y, r = e.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(y - r), 0.0, places=8)

    def test_update(self):
        p = Pacejka94()
        p.inputs[0] = 6.0
        p.inputs[1] = 12.0
        p.inputs[2] = 4500.0
        p.inputs[3] = 1.5

        p.update(0.0)

        self.assertTrue(np.isfinite(p.outputs[0]))
        self.assertTrue(np.isfinite(p.outputs[1]))

    def test_constant_camber_input(self):
        p = Pacejka94()

        slip_angle_deg = 4.0
        slip_ratio_pct = 9.0
        fz_N = 4300.0
        camber_deg = 2.0

        p.inputs[0] = slip_angle_deg
        p.inputs[1] = slip_ratio_pct
        p.inputs[2] = fz_N
        p.inputs[3] = camber_deg

        p.update(0.0)

        fz_kN = fz_N / 1000.0
        fx_ref = _pacejka94_fx_ref(slip_ratio_pct, fz_kN)
        fy_ref = _pacejka94_fy_ref(slip_angle_deg, fz_kN, camber_deg=camber_deg)

        self.assertAlmostEqual(p.outputs[0], fx_ref, places=8)
        self.assertAlmostEqual(p.outputs[1], fy_ref, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
