"""Tests for the kinematic bicycle model block."""

import numpy as np
import pytest

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Scope

from pathsim_vehicle import KinematicBicycle, BicycleParameters


@pytest.fixture()
def params():
    return BicycleParameters(l_f=1.105, l_r=1.738)


class TestKinematicBicycleODE:
    """Test the ODE right-hand side directly."""

    def test_straight_line(self, params):
        """Zero steering → straight-line motion along heading."""
        bike = KinematicBicycle(params, v0=10.0)
        state = np.array([0.0, 0.0, 0.0, 10.0])
        u = np.array([0.0, 0.0])  # delta_f=0, a=0
        f = bike.func_dyn(state, u, 0.0)

        # dx = v*cos(0) = 10, dy = v*sin(0) = 0, dpsi = 0, dv = 0
        np.testing.assert_allclose(f, [10.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_acceleration(self, params):
        """Pure acceleration from rest → dv = a."""
        bike = KinematicBicycle(params, v0=0.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        u = np.array([0.0, 2.0])  # delta_f=0, a=2
        f = bike.func_dyn(state, u, 0.0)

        assert f[3] == pytest.approx(2.0)  # dv = a
        assert f[0] == pytest.approx(0.0)  # dx = 0 (v=0)

    def test_steering_produces_yaw(self, params):
        """Nonzero steering at nonzero speed → nonzero yaw rate."""
        bike = KinematicBicycle(params, v0=5.0)
        state = np.array([0.0, 0.0, 0.0, 5.0])
        u = np.array([0.3, 0.0])  # delta_f = 0.3 rad
        f = bike.func_dyn(state, u, 0.0)

        assert abs(f[2]) > 0  # dpsi != 0


class TestKinematicBicycleSimulation:
    """Integration tests using PathSim simulation.

    Scope.read() returns (time, data) where data has shape (n_channels, n_steps).
    Channel order: 0=x, 1=y, 2=psi, 3=v.
    """

    def test_straight_line_integration(self, params):
        """Constant speed, zero steering → straight line."""
        bike = KinematicBicycle(params, v0=10.0)
        src_delta = Source(lambda t: 0.0)
        src_a = Source(lambda t: 0.0)
        sco = Scope(labels=["x", "y", "psi", "v"])

        sim = Simulation(
            [src_delta, src_a, bike, sco],
            [
                Connection(src_delta, bike),
                Connection(src_a, bike[1]),
                Connection(bike, sco),
                Connection(bike[1], sco[1]),
                Connection(bike[2], sco[2]),
                Connection(bike[3], sco[3]),
            ],
            dt=0.01,
        )
        sim.run(1.0)

        _, data = sco.read()
        # After 1s at v=10 m/s: x ≈ 10, y ≈ 0
        assert data[0, -1] == pytest.approx(10.0, abs=0.1)  # x
        assert data[1, -1] == pytest.approx(0.0, abs=0.01)  # y

    def test_circular_trajectory(self, params):
        """Constant steering + constant speed → circular arc."""
        bike = KinematicBicycle(params, v0=5.0)
        delta_f = 0.2  # rad
        src_delta = Source(lambda t: delta_f)
        src_a = Source(lambda t: 0.0)
        sco = Scope(labels=["x", "y", "psi", "v"])

        sim = Simulation(
            [src_delta, src_a, bike, sco],
            [
                Connection(src_delta, bike),
                Connection(src_a, bike[1]),
                Connection(bike, sco),
                Connection(bike[1], sco[1]),
                Connection(bike[2], sco[2]),
                Connection(bike[3], sco[3]),
            ],
            dt=0.01,
        )
        sim.run(2.0)

        _, data = sco.read()
        # Should have turned — psi should change significantly
        assert abs(data[2, -1]) > 0.1  # psi changed
        # y should be nonzero (turning)
        assert abs(data[1, -1]) > 0.1

    def test_zero_velocity_no_motion(self, params):
        """Zero velocity, zero acceleration → no motion."""
        bike = KinematicBicycle(params, v0=0.0)
        src_delta = Source(lambda t: 0.3)  # steering applied but v=0
        src_a = Source(lambda t: 0.0)
        sco = Scope(labels=["x", "y", "psi", "v"])

        sim = Simulation(
            [src_delta, src_a, bike, sco],
            [
                Connection(src_delta, bike),
                Connection(src_a, bike[1]),
                Connection(bike, sco),
                Connection(bike[1], sco[1]),
                Connection(bike[2], sco[2]),
                Connection(bike[3], sco[3]),
            ],
            dt=0.01,
        )
        sim.run(0.5)

        _, data = sco.read()
        assert data[0, -1] == pytest.approx(0.0, abs=1e-10)  # no x motion
        assert data[1, -1] == pytest.approx(0.0, abs=1e-10)  # no y motion
