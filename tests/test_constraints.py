"""Tests for bicycle constraint functions."""

import numpy as np
import pytest

from pathsim_vehicle.constraints import (
    BicycleConstraints,
    clip_steering,
    clip_acceleration,
)


@pytest.fixture()
def c():
    """Default constraints from Table II."""
    return BicycleConstraints()


class TestBicycleConstraints:

    def test_defaults_symmetric_steering(self, c):
        assert c.delta_min == pytest.approx(-c.delta_max)

    def test_defaults_from_paper(self, c):
        # 37° ≈ 0.6458 rad
        assert c.delta_max == pytest.approx(0.6458, abs=0.001)
        assert c.a_max == pytest.approx(1.0)
        assert c.a_min == pytest.approx(-1.5)


class TestClipSteering:

    def test_within_bounds(self, c):
        assert clip_steering(0.3, c) == pytest.approx(0.3)

    def test_clip_positive(self, c):
        assert clip_steering(2.0, c) == pytest.approx(c.delta_max)

    def test_clip_negative(self, c):
        assert clip_steering(-2.0, c) == pytest.approx(c.delta_min)

    def test_array_input(self, c):
        vals = np.array([-2.0, 0.0, 2.0])
        result = clip_steering(vals, c)
        np.testing.assert_allclose(result, [c.delta_min, 0.0, c.delta_max])


class TestClipAcceleration:

    def test_within_bounds(self, c):
        assert clip_acceleration(0.5, c) == pytest.approx(0.5)

    def test_clip_positive(self, c):
        assert clip_acceleration(5.0, c) == pytest.approx(c.a_max)

    def test_clip_negative(self, c):
        assert clip_acceleration(-5.0, c) == pytest.approx(c.a_min)

    def test_array_input(self, c):
        vals = np.array([-10.0, 0.0, 10.0])
        result = clip_acceleration(vals, c)
        np.testing.assert_allclose(result, [c.a_min, 0.0, c.a_max])
