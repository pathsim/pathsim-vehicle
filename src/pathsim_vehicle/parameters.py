"""Vehicle parameter dataclasses for bicycle models.

Based on: Kong, J., Pfeiffer, M., Schildbach, G., and Borrelli, F.
"Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design",
IEEE IV, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BicycleParameters:
    """Parameters for the kinematic bicycle model (Eq. 1 in Kong et al.).

    Only two geometric parameters are needed: the distances from the
    center of gravity to the front and rear axles.

    Parameters
    ----------
    l_f : float
        Distance from center of gravity to front axle [m].
    l_r : float
        Distance from center of gravity to rear axle [m].
    """

    l_f: float
    l_r: float

    @property
    def wheelbase(self) -> float:
        """Total wheelbase l_f + l_r [m]."""
        return self.l_f + self.l_r


@dataclass
class DynamicBicycleParameters(BicycleParameters):
    """Extended parameters for the dynamic bicycle model (Eq. 2 in Kong et al.).

    Adds mass, yaw inertia, and linear tire cornering stiffnesses
    to the kinematic parameters.

    Parameters
    ----------
    l_f : float
        Distance from center of gravity to front axle [m].
    l_r : float
        Distance from center of gravity to rear axle [m].
    m : float
        Vehicle mass [kg].
    I_z : float
        Yaw moment of inertia [kg·m²].
    C_af : float
        Front axle cornering stiffness (both tires combined) [N/rad].
    C_ar : float
        Rear axle cornering stiffness (both tires combined) [N/rad].
    """

    m: float = 0.0
    I_z: float = 0.0
    C_af: float = 0.0
    C_ar: float = 0.0


def hyundai_azera() -> DynamicBicycleParameters:
    """Hyundai Azera test vehicle from Kong et al. IV 2015.

    Wheelbase: 2.843 m, l_f = 1.105 m, l_r = 1.738 m.
    Mass and inertia values are representative estimates for a
    mid-size sedan (~1600 kg). Cornering stiffness values are
    typical for a passenger car axle (~80,000 N/rad total, ~40,000 per tire).
    """
    return DynamicBicycleParameters(
        l_f=1.105,
        l_r=1.738,
        m=1600.0,
        I_z=2250.0,
        C_af=126000.0,  # proportional to rear load fraction l_r/(l_f+l_r)
        C_ar=80000.0,   # proportional to front load fraction l_f/(l_f+l_r)
    )
