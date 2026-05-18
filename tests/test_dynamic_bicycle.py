"""Tests for the dynamic bicycle model block."""

import numpy as np
import pytest

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Scope

from pathsim_vehicle import DynamicBicycle, DynamicBicycleParameters


@pytest.fixture()
def params():
    return DynamicBicycleParameters(
        l_f=1.105,
        l_r=1.738,
        m=1600.0,
        I_z=2250.0,
        C_af=80000.0,
        C_ar=80000.0,
    )


class TestDynamicBicycleODE:
    """Test the ODE right-hand side directly."""

    def test_straight_line(self, params):
        """Zero steering, straight-line cruise → no lateral forces."""
        bike = DynamicBicycle(params, vx0=10.0)
        state = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0])  # delta_f=0, a_x=0
        f = bike.func_dyn(state, u, 0.0)

        # dvx = 0, dvy = 0, dpsi = 0
        assert f[0] == pytest.approx(0.0, abs=1e-10)
        assert f[1] == pytest.approx(0.0, abs=1e-10)
        assert f[2] == pytest.approx(0.0, abs=1e-10)
        # dX = vx*cos(0) = 10
        assert f[4] == pytest.approx(10.0, abs=1e-10)
        # dY = vx*sin(0) = 0
        assert f[5] == pytest.approx(0.0, abs=1e-10)

    def test_acceleration(self, params):
        """Pure longitudinal acceleration."""
        bike = DynamicBicycle(params, vx0=5.0)
        state = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([0.0, 2.0])  # a_x = 2 m/s²
        f = bike.func_dyn(state, u, 0.0)

        assert f[0] == pytest.approx(2.0, abs=1e-10)  # dvx = a_x


class TestDynamicBicycleSimulation:
    """Integration tests using PathSim simulation.

    Scope.read() returns (time, data) where data has shape (n_channels, n_steps).
    Channel order: 0=vx, 1=vy, 2=psi, 3=psi_dot, 4=X, 5=Y.
    """

    def test_straight_line_integration(self, params):
        """Constant speed, zero steering → straight line."""
        bike = DynamicBicycle(params, vx0=10.0)
        src_delta = Source(lambda t: 0.0)
        src_a = Source(lambda t: 0.0)
        sco = Scope(labels=["vx", "vy", "psi", "psi_dot", "X", "Y"])

        sim = Simulation(
            [src_delta, src_a, bike, sco],
            [
                Connection(src_delta, bike),
                Connection(src_a, bike[1]),
                Connection(bike, sco),
                Connection(bike[1], sco[1]),
                Connection(bike[2], sco[2]),
                Connection(bike[3], sco[3]),
                Connection(bike[4], sco[4]),
                Connection(bike[5], sco[5]),
            ],
            dt=0.01,
        )
        sim.run(1.0)

        _, data = sco.read()
        # X ≈ 10 m after 1s at 10 m/s
        assert data[4, -1] == pytest.approx(10.0, abs=0.1)
        # Y ≈ 0
        assert data[5, -1] == pytest.approx(0.0, abs=0.01)

    def test_steering_produces_lateral_motion(self, params):
        """Steering at speed → lateral motion appears."""
        bike = DynamicBicycle(params, vx0=10.0)
        src_delta = Source(lambda t: 0.1)  # constant small steer
        src_a = Source(lambda t: 0.0)
        sco = Scope(labels=["vx", "vy", "psi", "psi_dot", "X", "Y"])

        sim = Simulation(
            [src_delta, src_a, bike, sco],
            [
                Connection(src_delta, bike),
                Connection(src_a, bike[1]),
                Connection(bike, sco),
                Connection(bike[1], sco[1]),
                Connection(bike[2], sco[2]),
                Connection(bike[3], sco[3]),
                Connection(bike[4], sco[4]),
                Connection(bike[5], sco[5]),
            ],
            dt=0.01,
        )
        sim.run(2.0)

        _, data = sco.read()
        # Should have turned — Y nonzero
        assert abs(data[5, -1]) > 0.5
        # Heading changed
        assert abs(data[2, -1]) > 0.05

    def test_low_speed_safe(self, params):
        """At very low speed, model should not blow up (slip guard)."""
        bike = DynamicBicycle(params, vx0=0.1)
        src_delta = Source(lambda t: 0.2)
        src_a = Source(lambda t: 0.0)
        sco = Scope(labels=["vx", "vy", "psi", "psi_dot", "X", "Y"])

        sim = Simulation(
            [src_delta, src_a, bike, sco],
            [
                Connection(src_delta, bike),
                Connection(src_a, bike[1]),
                Connection(bike, sco),
                Connection(bike[1], sco[1]),
                Connection(bike[2], sco[2]),
                Connection(bike[3], sco[3]),
                Connection(bike[4], sco[4]),
                Connection(bike[5], sco[5]),
            ],
            dt=0.01,
        )
        sim.run(0.5)

        _, data = sco.read()
        # All values should be finite (no NaN/Inf from division by zero)
        assert np.all(np.isfinite(data))
