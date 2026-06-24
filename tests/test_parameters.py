"""Tests for vehicle parameter dataclasses."""

import pytest

from pathsim_vehicle.parameters import (
    BicycleParameters,
    DynamicBicycleParameters,
    hyundai_azera,
)


class TestBicycleParameters:

    def test_construction(self):
        p = BicycleParameters(l_f=1.0, l_r=1.5)
        assert p.l_f == 1.0
        assert p.l_r == 1.5

    def test_wheelbase(self):
        p = BicycleParameters(l_f=1.0, l_r=1.5)
        assert p.wheelbase == pytest.approx(2.5)


class TestDynamicBicycleParameters:

    def test_inherits_bicycle(self):
        p = DynamicBicycleParameters(
            l_f=1.0, l_r=1.5, m=1500.0, I_z=2000.0, C_af=80000.0, C_ar=80000.0
        )
        assert isinstance(p, BicycleParameters)
        assert p.wheelbase == pytest.approx(2.5)

    def test_dynamic_fields(self):
        p = DynamicBicycleParameters(
            l_f=1.0, l_r=1.5, m=1500.0, I_z=2000.0, C_af=80000.0, C_ar=80000.0
        )
        assert p.m == 1500.0
        assert p.I_z == 2000.0
        assert p.C_af == 80000.0
        assert p.C_ar == 80000.0


class TestHyundaiAzera:

    def test_returns_dynamic_params(self):
        p = hyundai_azera()
        assert isinstance(p, DynamicBicycleParameters)

    def test_wheelbase(self):
        p = hyundai_azera()
        assert p.wheelbase == pytest.approx(2.843)

    def test_axle_distances(self):
        p = hyundai_azera()
        assert p.l_f == pytest.approx(1.105)
        assert p.l_r == pytest.approx(1.738)

    def test_mass_positive(self):
        p = hyundai_azera()
        assert p.m > 0
        assert p.I_z > 0
        assert p.C_af > 0
        assert p.C_ar > 0
