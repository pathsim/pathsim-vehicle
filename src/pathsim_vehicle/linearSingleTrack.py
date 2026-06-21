#########################################################################################
##
##                  Linear single-track Block
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.ode import ODE


# BLOCK Definitions ================================================================================

class LinearSingleTrack(ODE):
    """Linearized single-track (bicycle) vehicle model with external
    longitudinal velocity input.

    The model is valid for small slip angles and lateral accelerations
    within the linear tire range (roughly :math:`a_y < 4\\,\\mathrm{m/s^2}`
    on dry asphalt for typical passenger cars) at moderate, slowly varying
    forward speed. Near standstill the slip-angle kinematics are singular
    in :math:`v_x`; the implementation replaces it with the smooth norm
    :math:`\\sqrt{v_x^2 + v_{x,\\mathrm{eps}}^2}`, so the model is not
    physically meaningful below :math:`v_{x,\\mathrm{eps}}`.

    The equations of the ``LinearSingleTrack`` block are derived from the
    nonlinear, force-driven single-track model with the equations of motion
    (body frame, ISO 8855)

    .. math::

        \\begin{aligned}
        m\\,\\dot v_x &= F_{x,f}^{b} + F_{x,r}^{b} + m\\,v_y\\,r, \\\\
        m\\,\\dot v_y &= F_{y,f}^{b} + F_{y,r}^{b} - m\\,v_x\\,r, \\\\
        I_z\\,\\dot r &= l_f\\,F_{y,f}^{b} - l_r\\,F_{y,r}^{b},
        \\end{aligned}

    where the body-frame axle forces are obtained by rotating the
    tire-frame forces through the steer angles,

    .. math::

        \\begin{aligned}
        F_{x,f}^{b} &= F_{x,f}\\cos\\delta   - F_{y,f}\\sin\\delta, \\\\
        F_{y,f}^{b} &= F_{x,f}\\sin\\delta   + F_{y,f}\\cos\\delta, \\\\
        F_{x,r}^{b} &= F_{x,r}\\cos\\delta_r - F_{y,r}\\sin\\delta_r, \\\\
        F_{y,r}^{b} &= F_{x,r}\\sin\\delta_r + F_{y,r}\\cos\\delta_r,
        \\end{aligned}

    and the tire slip angles are

    .. math::

        \\alpha_f = \\delta - \\arctan\\frac{v_y + l_f\\,r}{v_x},
        \\qquad
        \\alpha_r = \\delta_r - \\arctan\\frac{v_y - l_r\\,r}{v_x}.

    The model is linearized about steady straight-line driving at speed
    :math:`v_x` using the following assumptions:

    - **Quasi-steady longitudinal motion**: :math:`\\dot v_x \\approx 0`.
      Under this assumption the longitudinal equation of motion is dropped
      and :math:`v_x` becomes an external input.

    - **Front-axle steering only**: :math:`\\delta_r = 0`.

    - **Small angles**: steering angle, sideslip, and tire slip angles are
      small, :math:`\\delta, \\beta, \\alpha_f, \\alpha_r \\ll 1`. With
      :math:`\\cos\\delta \\approx 1` and :math:`\\sin\\delta \\approx \\delta`
      the force rotation reduces to

      .. math::

          F_{y,f}^{b} \\approx F_{y,f},
          \\qquad
          F_{y,r}^{b} \\approx F_{y,r},

      and with :math:`\\arctan x \\approx x` the slip angles become

      .. math::

          \\alpha_f = \\delta - \\frac{v_y + l_f\\,r}{v_x},
          \\qquad
          \\alpha_r = -\\,\\frac{v_y - l_r\\,r}{v_x}.

    - **Linear tire model**:

      .. math::

          F_{y,f} = C_{F\\alpha,f}\\,\\alpha_f,
          \\qquad
          F_{y,r} = C_{F\\alpha,r}\\,\\alpha_r.

    Substituting these into the lateral and yaw equations of motion and
    writing :math:`C_f \\equiv C_{F\\alpha,f}` and
    :math:`C_r \\equiv C_{F\\alpha,r}` yields the linear state equations

    .. math::

        \\begin{aligned}
        \\dot v_y &= -\\frac{C_f + C_r}{m\\,v_x}\\,v_y
          + \\left( \\frac{C_r l_r - C_f l_f}{m\\,v_x} - v_x \\right) r
          + \\frac{C_f}{m}\\,\\delta, \\\\
        \\dot r &= \\frac{C_r l_r - C_f l_f}{I_z\\,v_x}\\,v_y
          - \\frac{C_f l_f^{2} + C_r l_r^{2}}{I_z\\,v_x}\\,r
          + \\frac{C_f l_f}{I_z}\\,\\delta.
        \\end{aligned}

    The pose kinematics are appended in their exact nonlinear form:

    .. math::

        \\dot\\psi = r,
        \\qquad
        \\dot X = v_x\\cos\\psi - v_y\\sin\\psi,
        \\qquad
        \\dot Y = v_x\\sin\\psi + v_y\\cos\\psi.

    
    Input Ports
    -----------
    delta : float
        front-axle steering angle [rad]
    v_x : float
        longitudinal velocity [m/s]

    Output Ports
    ------------
    v_y : float
        lateral velocity [m/s]
    r : float
        yaw rate [rad/s]
    psi : float
        yaw angle [rad]
    X : float
        vehicle position along the global X-axis [m]
    Y : float
        vehicle position along the global Y-axis [m]


    Parameters
    ----------
    m : float
        Vehicle mass [kg].
    I_z : float
        Yaw moment of inertia [kg m^2].
    l_f : float
        Distance from CG to front axle [m].
    l_r : float
        Distance from CG to rear axle [m].
    C_Falpha_f : float
        Front-axle cornering stiffness [N/rad], > 0.
    C_Falpha_r : float
        Rear-axle cornering stiffness [N/rad], > 0.
    initial_value : array_like, optional
        Initial state vector ``[v_y, r, psi, X, Y]``


    """

    # port labels for semantic access
    input_port_labels  = {"delta": 0, "v_x": 1}
    output_port_labels = {"v_y": 0, "r": 1, "psi": 2, "X": 3, "Y": 4}

    def __init__(self, m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4,
                 C_Falpha_f=80000.0, C_Falpha_r=80000.0, v_x_eps = 0.5, initial_value=None):

        # vehicle parameters
        self.m = m
        self.I_z = I_z
        self.l_f = l_f
        self.l_r = l_r
        self.C_Falpha_f = C_Falpha_f
        self.C_Falpha_r = C_Falpha_r
        self.v_x_eps = v_x_eps

        
        if initial_value is None:
            initial_value = np.zeros(5)

        super().__init__(
            func=self._func_dyn,
            initial_value=np.asarray(initial_value, dtype=float),
            jac=self._jac_dyn,
            )


    def _func_dyn(self, x, u, t):
        """Right-hand side of the linear single-track ODEs.

        Parameters
        ----------
        x : array[float]
            State vector ``[v_y, r, psi, X, Y]``.
        u : array[float]
            Input vector ``[delta, v_x]``.
        t : float
            Time.

        Returns
        -------
        dxdt : array[float]
            State derivative vector.
        """
        v_y, r, psi, X, Y = x
        delta, v_x = u[0], u[1]

        # sign-preserving singularity guard for the slip-angle denominator
        v_x_safe = np.sqrt(v_x**2 + self.v_x_eps**2)

        # linearised slip angles (small-angle: tan(alpha) ~ alpha)
        alpha_f = delta - (v_y + self.l_f * r) / v_x_safe
        alpha_r =       - (v_y - self.l_r * r) / v_x_safe

        # linear tyre lateral forces
        F_y_f = self.C_Falpha_f * alpha_f
        F_y_r = self.C_Falpha_r * alpha_r

        # equations of motion
        dv_y = (F_y_f + F_y_r) / self.m - v_x * r
        dr   = (self.l_f * F_y_f - self.l_r * F_y_r) / self.I_z
        dpsi = r
        dX   = v_x * np.cos(psi) - v_y * np.sin(psi)
        dY   = v_x * np.sin(psi) + v_y * np.cos(psi)

        return np.array([dv_y, dr, dpsi, dX, dY])


    def _jac_dyn(self, x, u, t):
        """Analytic state Jacobian df/dx of the linear single-track equations.
        """
        v_y, r, psi = x[0], x[1], x[2]
        v_x = u[1]
        v_x_safe = np.sqrt(v_x**2 + self.v_x_eps**2)

        J = np.zeros((5, 5))
        J[0, 0] = (-self.C_Falpha_f - self.C_Falpha_r)/(self.m*v_x_safe)
        J[0, 1] = (-self.C_Falpha_f*self.l_f + self.C_Falpha_r*self.l_r - self.m*v_x*v_x_safe)/(self.m*v_x_safe)
        J[1, 0] = (-self.C_Falpha_f*self.l_f + self.C_Falpha_r*self.l_r)/(self.I_z*v_x_safe)
        J[1, 1] = (-self.C_Falpha_f*self.l_f**2 - self.C_Falpha_r*self.l_r**2)/(self.I_z*v_x_safe)
        J[2, 1] = 1
        J[3, 0] = -np.sin(psi)
        J[3, 2] = -v_x*np.sin(psi) - v_y*np.cos(psi)
        J[4, 0] = np.cos(psi)
        J[4, 2] = v_x*np.cos(psi) - v_y*np.sin(psi)
        return J
