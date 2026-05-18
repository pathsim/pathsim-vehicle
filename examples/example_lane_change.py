#########################################################################################
##
##              PathSim-Vehicle: Kinematic Bicycle Lane Change Example
##
##  Simulates a double lane change maneuver comparing the kinematic and dynamic
##  bicycle models from Kong et al. (IEEE IV, 2015).
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Scope
from pathsim.solvers import RKCK54

from pathsim_vehicle import (
    KinematicBicycle,
    DynamicBicycle,
    hyundai_azera,
)


# VEHICLE PARAMETERS ===================================================================

params = hyundai_azera()

# Initial conditions
v0 = 15.0  # Highway speed [m/s] (~54 km/h)


# STEERING INPUT ========================================================================

# Double lane change: sinusoidal steering profile
delta_max = np.radians(5.0)   # Peak steering angle [rad]
t_start = 1.0                 # Maneuver start time [s]
t_duration = 3.0              # Maneuver duration [s]
t_end = t_start + t_duration  # Maneuver end time [s]

def steering_input(t):
    """Double lane change steering profile.

    A full sine wave that steers right then left, returning to straight.
    """
    if t < t_start or t > t_end:
        return 0.0
    phase = 2.0 * np.pi * (t - t_start) / t_duration
    return delta_max * np.sin(phase)


# KINEMATIC MODEL SETUP ================================================================

kin = KinematicBicycle(params, v0=v0)
src_delta_k = Source(steering_input)
src_accel_k = Source(lambda t: 0.0)  # Constant speed

sco_kin = Scope(labels=["x [m]", "y [m]", "ψ [rad]", "v [m/s]"])

blocks_kin = [src_delta_k, src_accel_k, kin, sco_kin]

connections_kin = [
    Connection(src_delta_k, kin["delta_f"]),
    Connection(src_accel_k, kin["a"]),
    Connection(kin["x"],   sco_kin[0]),
    Connection(kin["y"],   sco_kin[1]),
    Connection(kin["psi"], sco_kin[2]),
    Connection(kin["v"],   sco_kin[3]),
]

Sim_kin = Simulation(blocks_kin, connections_kin, Solver=RKCK54, dt=0.01)


# DYNAMIC MODEL SETUP ==================================================================

dyn = DynamicBicycle(params, vx0=v0)
src_delta_d = Source(steering_input)
src_accel_d = Source(lambda t: 0.0)

sco_dyn = Scope(labels=["vx [m/s]", "vy [m/s]", "ψ [rad]", "ψ̇ [rad/s]", "X [m]", "Y [m]"])

blocks_dyn = [src_delta_d, src_accel_d, dyn, sco_dyn]

connections_dyn = [
    Connection(src_delta_d, dyn["delta_f"]),
    Connection(src_accel_d, dyn["a_x"]),
    Connection(dyn["vx"],      sco_dyn[0]),
    Connection(dyn["vy"],      sco_dyn[1]),
    Connection(dyn["psi"],     sco_dyn[2]),
    Connection(dyn["psi_dot"], sco_dyn[3]),
    Connection(dyn["X"],       sco_dyn[4]),
    Connection(dyn["Y"],       sco_dyn[5]),
]

Sim_dyn = Simulation(blocks_dyn, connections_dyn, Solver=RKCK54, dt=0.01)


# SIMULATION ============================================================================

T_sim = 6.0  # Total simulation time [s]


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run both simulations
    Sim_kin.run(T_sim)
    Sim_dyn.run(T_sim)

    # Read results
    t_k, data_k = sco_kin.read()
    t_d, data_d = sco_dyn.read()

    # ---- Plot ----

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Kinematic vs Dynamic Bicycle Model — Double Lane Change", fontsize=14)

    # (0,0) XY trajectory
    ax = axes[0, 0]
    ax.plot(data_k[0], data_k[1], label="Kinematic", lw=2)
    ax.plot(data_d[4], data_d[5], "--", label="Dynamic", lw=2)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory (top view)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # (0,1) Heading angle
    ax = axes[0, 1]
    ax.plot(t_k, np.degrees(data_k[2]), label="Kinematic ψ", lw=2)
    ax.plot(t_d, np.degrees(data_d[2]), "--", label="Dynamic ψ", lw=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heading ψ [°]")
    ax.set_title("Heading Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Steering input
    ax = axes[1, 0]
    t_steer = np.linspace(0, T_sim, 500)
    delta_vals = np.array([steering_input(t) for t in t_steer])
    ax.plot(t_steer, np.degrees(delta_vals), "k-", lw=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Steering δ_f [°]")
    ax.set_title("Steering Input")
    ax.grid(True, alpha=0.3)

    # (1,1) Lateral displacement
    ax = axes[1, 1]
    ax.plot(t_k, data_k[1], label="Kinematic y", lw=2)
    ax.plot(t_d, data_d[5], "--", label="Dynamic Y", lw=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Lateral displacement [m]")
    ax.set_title("Lateral Position vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
