import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def cross_product_matrix(vec):
    cpm = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    return cpm


@jit(nopython=True)
def matrix_vector(x, y):
    return x @ y.T


class Surface:
    def __init__(
        self,
        x_cm_surf,
        c,
        b,
        n,
        CL0,
        e,
        i,
        CD0,
        CDa,
        a0,
        CM0,
        CMa,
        mass_props,
        number,
        x_cm,
    ):

        # self.name = name
        self.number = number
        self.mass_props = mass_props

        self.x_cm_surf = x_cm_surf
        self.x_cm_aircraft = x_cm

        if c is None:
            self.Cref = mass_props[1]
        else:
            self.Cref = c

        if b is None:
            self.Bref = max(mass_props[2], mass_props[3])
        else:
            self.Bref = b

        self.n_b = n
        self.CL_0 = CL0
        self.e_surf = e
        self.i = i
        self.CD_0 = CD0
        self.CD_a = CDa
        self.a0 = a0
        self.CM_0 = CM0
        self.CM_a = CMa

        # Constants
        self.dCLdu = 3
        self.rho = 1.225
        self.q_bar = None
        # Calculated Parameters
        self.Sref = None  # surface area
        self.AR = None  # surface aspect ratio
        self.CL_a = None
        self.Cu = None
        self.Su = None
        self.SuS = None
        self.dx_mat = None
        self.dx_cm = None
        self.get_calculated_parameters()

        # Input Values
        self.v_B_B_BA = None
        self.v_inf = None
        self.C_BW = None
        self.u = None
        self.w_B_B_BE = None

        # Intermediate Values
        self.a_surf = None
        self.CD = None
        self.CL = None
        self.CM = None
        self.L_surf_W = None
        self.D_surf_W = None
        self.M_surf_W = None

        # Output Values
        self.F_surf_B = None
        self.M_surf_B = None

    def get_calculated_parameters(self):
        if self.x_cm_surf is None:
            self.calc_surface_location()
        self.dx_mat = cross_product_matrix(self.x_cm_surf - self.x_cm_aircraft)
        self.calc_surface_area()
        self.calc_aspect_ratio()
        self.calc_lift_curve_slope()
        self.calc_Cu()
        self.calc_Su()
        self.calc_SuS()
        self.dx_cm = self.x_cm_surf - self.x_cm_aircraft

    def calc_surface_location(self):
        self.x_cm_surf = np.array(
            [
                self.mass_props[4] + (1 / 4) * self.mass_props[1],
                self.mass_props[5],
                self.mass_props[6],
            ]
        )

    def calc_surface_area(self):
        self.Sref = self.Cref * self.Bref

    def calc_aspect_ratio(self):
        self.AR = self.Bref**2 / self.Sref

    def calc_lift_curve_slope(self):
        self.CL_a = 2 * np.pi * self.AR / (2 + self.AR)

    def calc_Cu(self):
        self.Cu = 0.2 * self.Cref

    def calc_Su(self):
        self.Su = self.Cu * self.Bref

    def calc_SuS(self):
        self.SuS = self.Su / self.Sref

    def forces_and_moments(self, v_B_B_BA, v_inf, C_BW, u, w_B_B_BE):
        self.v_inf = v_inf
        self.q_bar = (1 / 2) * self.rho * self.v_inf**2
        self.v_B_B_BA = v_B_B_BA
        self.C_BW = C_BW
        self.u = u
        self.w_B_B_BE = w_B_B_BE
        self.calc_surface_AOA()
        self.calc_surface_forces()
        self.calc_surface_moments()

    def calc_surface_forces(self):
        self.calc_lift()
        self.calc_drag()
        F_surf_W = np.array([-self.D_surf_W, 0, 0]) + self.L_surf_W * self.n_b
        self.F_surf_B = matrix_vector(self.C_BW, F_surf_W)

    def calc_surface_moments(self):
        self.calc_moment()

        C_Moment = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])

        self.M_surf_B = ((self.M_surf_W * C_Moment) @ self.n_b.T) + (
            matrix_vector(self.dx_mat, self.F_surf_B)
        )

    def calc_surface_AOA(self):
        w_B_B_BE_mat = cross_product_matrix(self.w_B_B_BE)
        v_B_B_SnA = self.v_B_B_BA + matrix_vector(w_B_B_BE_mat, self.dx_cm)
        self.a_surf = self.i - np.arctan2(np.dot(v_B_B_SnA, self.n_b), v_B_B_SnA[0])

    def calc_lift_coefficient(self):
        self.CL = self.CL_0 + self.CL_a * self.a_surf + self.SuS * self.dCLdu * self.u

    def calc_lift(self):
        self.calc_lift_coefficient()
        self.L_surf_W = self.CL * self.q_bar * self.Sref

    def calc_drag_coefficient(self):
        CD_temp = (
            self.CD_0
            + self.CD_a * (self.a_surf - self.a0) ** 2
            + self.CL**2 / (np.pi * self.e_surf * self.AR)
        )

        self.CD = max(CD_temp, self.CD_0)

    def calc_drag(self):
        self.calc_drag_coefficient()
        self.D_surf_W = self.CD * self.q_bar * self.Sref

    def calc_moment_coefficient(self):
        self.CM = self.CM_0 + self.CM_a * self.a_surf

    def calc_moment(self):
        self.calc_moment_coefficient()
        self.M_surf_W = self.CM * self.q_bar * self.Sref * self.Cref

    def get_surface_forces(self):
        return self.F_surf_B

    def get_surface_moments(self):
        return self.M_surf_B


#
# ap_spec = [
#     ('mass_props', float),
#     ('surfs', ),
#     ('J'),
#     ('J_inv'),
#     ('mass'),
#
# ]
#
# @jitclass(ap_spec)


class Aero_Parameters:
    def __init__(self):
        self.mass_props = np.load("simulation/mass_props.npy")
        self.surfs = None
        self.x_cm = None
        self.J = None
        self.J_inv = None
        self.mass = None

        self.mass_properties()
        self.initialize_surfaces()

    def initialize_surfaces(self):
        # Location of Surface 3
        x_s3 = np.array([-0.76, 0, -0.09])

        # Chord of Surface 3
        c_s3 = 0.08

        # Span of Surface 3
        b_s3 = 0.08

        # Normal Vectors
        n_s2 = np.array([0, 0, -1])
        n_s3 = np.array([0, 1, 0])
        n_s4 = np.array([0, 0, -1])
        n_s5 = np.array([0, 0, -1])

        # Lift Coefficient @ 0 AOA
        CL0_s2 = 0
        CL0_s3 = 0
        CL0_s4 = 0.05
        CL0_s5 = 0.05

        # Oswald Efficiency Factor
        e_s2 = 0.8
        e_s3 = 0.8
        e_s4 = 0.9
        e_s5 = 0.9

        # Mounting Incidence Angle
        i_s2 = 0
        i_s3 = 0
        i_s4 = 0.05
        i_s5 = 0.05

        # Minimum Drag Profile
        CD0_s2 = 0.01
        CD0_s3 = 0.01
        CD0_s4 = 0.01
        CD0_s5 = 0.01

        # Drag Coefficient
        CDa_s2 = 1
        CDa_s3 = 1
        CDa_s4 = 1
        CDa_s5 = 1

        # Minimum Drag AOA
        a0_s2 = 0
        a0_s3 = 0
        a0_s4 = 0.05
        a0_s5 = 0.05

        # Minimum Moment Coefficient
        CM0_s2 = 0
        CM0_s3 = 0
        CM0_s4 = -0.05
        CM0_s5 = -0.05

        # Moment Coefficient
        CMa_s2 = 0
        CMa_s3 = 0
        CMa_s4 = 0
        CMa_s5 = 0

        # Aero Control Surface Calculations
        mass_props_ac = np.array(
            [
                [90, 0.1, 0.48, 0.01, -0.23, 0.44, 0],
                [90, 0.1, 0.48, 0.01, -0.23, -0.44, 0],
                [13, 0.075, 0.35, 0.002, -0.76, 0, 0.16],
            ]
        )

        # Vertical Stabilizer but ASW28 has rudder so this doesn't need to
        # be separated
        np.array([[0, 0.08, 0.002, 0.18, -0.76, 0, -0.09]])

        np.array([[7.5 * np.pi / 180, 16 * np.pi / 180]])

        # S2 = Elevator = Surface 2
        s2_props = mass_props_ac[2, 0::]

        # X_cm_surf, c, b, n
        S2 = Surface(
            None,
            None,
            None,
            n_s2,
            CL0_s2,
            e_s2,
            i_s2,
            CD0_s2,
            CDa_s2,
            a0_s2,
            CM0_s2,
            CMa_s2,
            s2_props,
            "Elevator",
            2,
            self.x_cm,
        )

        # S3 = Rudder = Surface 3
        np.array([7.5 * np.pi / 180, 16 * np.pi / 180])

        S3 = Surface(
            x_s3,
            c_s3,
            b_s3,
            n_s3,
            CL0_s3,
            e_s3,
            i_s3,
            CD0_s3,
            CDa_s3,
            a0_s3,
            CM0_s3,
            CMa_s3,
            None,
            "Rudder",
            3,
            self.x_cm,
        )

        # S4 = Right Wing = Surface 4
        s4_props = mass_props_ac[0, 0::]

        S4 = Surface(
            None,
            None,
            None,
            n_s4,
            CL0_s4,
            e_s4,
            i_s4,
            CD0_s4,
            CDa_s4,
            a0_s4,
            CM0_s4,
            CMa_s4,
            s4_props,
            "Right Wing",
            4,
            self.x_cm,
        )
        # S5 = Left Wing = Surface 5
        s5_props = mass_props_ac[1, 0::]

        S5 = Surface(
            None,
            None,
            None,
            n_s5,
            CL0_s5,
            e_s5,
            i_s5,
            CD0_s5,
            CDa_s5,
            a0_s5,
            CM0_s5,
            CMa_s5,
            s5_props,
            "Left Wing",
            5,
            self.x_cm,
        )

        self.surfs = [S4, S5, S2, S3]

    def mass_properties(self):
        self.get_mass()
        self.get_center_of_mass()
        self.get_mass_moi()

    def get_mass(self):
        # Convert to kg
        self.mass_props[0::, 0] = self.mass_props[0::, 0] / 1000
        self.mass = np.sum(self.mass_props[0::, 0])

    def get_center_of_mass(self):

        # Splitting calculations up into 3 separate matrices
        x_moments = self.mass_props[0::, 0] * self.mass_props[0::, 4]
        y_moments = self.mass_props[0::, 0] * self.mass_props[0::, 5]
        z_moments = self.mass_props[0::, 0] * self.mass_props[0::, 6]

        # Combining arrays into one moment matrix
        componentMoments = np.vstack((x_moments, y_moments, z_moments)).T

        # Summing columns and dividing by mass
        self.x_cm = (sum(componentMoments) / self.mass).T

    def get_mass_moi(self):
        inertias = np.empty((10, 3))
        for i in range(len(self.mass_props[0::, 0])):
            if self.mass_props[i, 1] != 0:
                inertias[i, 0] = (
                    (1 / 12)
                    * self.mass_props[i, 0]
                    * (
                        np.square(self.mass_props[i, 2])
                        + np.square(self.mass_props[i, 3])
                    )
                )

                inertias[i, 1] = (
                    (1 / 12)
                    * self.mass_props[i, 0]
                    * (
                        np.square(self.mass_props[i, 3])
                        + np.square(self.mass_props[i, 1])
                    )
                )

                inertias[i, 2] = (
                    (1 / 12)
                    * self.mass_props[i, 0]
                    * (
                        np.square(self.mass_props[i, 1])
                        + np.square(self.mass_props[i, 2])
                    )
                )
            else:
                inertias[i, 0] = (
                    (1 / 12)
                    * self.mass_props[i, 0]
                    * (
                        np.square(self.mass_props[i, 2])
                        + np.square(self.mass_props[i, 3])
                    )
                )

        J = np.empty((3, 3))
        J[0, 0] = np.sum(inertias[0::, 0]) + np.sum(
            self.mass_props[0::, 0]
            * (
                np.square(self.mass_props[0::, 5] - self.x_cm[1])
                + np.square(self.mass_props[0::, 6] - self.x_cm[2])
            )
        )
        J[1, 1] = np.sum(inertias[0::, 1]) + np.sum(
            self.mass_props[0::, 0]
            * (
                np.square(self.mass_props[0::, 6] - self.x_cm[2])
                + np.square(self.mass_props[0::, 4] - self.x_cm[0])
            )
        )
        J[2, 2] = np.sum(inertias[0::, 2]) + np.sum(
            self.mass_props[0::, 0]
            * (
                np.square(self.mass_props[0::, 4] - self.x_cm[0])
                + np.square(self.mass_props[0::, 5] - self.x_cm[1])
            )
        )
        J[0, 1] = -np.sum(
            self.mass_props[0::, 0]
            * (self.mass_props[0::, 4] * self.mass_props[0::, 5])
        )
        J[0, 2] = -np.sum(
            self.mass_props[0::, 0]
            * (self.mass_props[0::, 6] * self.mass_props[0::, 4])
        )
        J[1, 2] = -np.sum(
            self.mass_props[0::, 0]
            * (self.mass_props[0::, 5] * self.mass_props[0::, 6])
        )

        J[0, 1] = -1 * J[0, 1]
        J[0, 2] = -1 * J[0, 2]
        J[1, 2] = -1 * J[1, 2]
        J[1, 0] = J[0, 1]
        J[2, 0] = J[0, 2]
        J[2, 1] = J[1, 2]

        self.J = J
        df = pd.DataFrame(J)
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            print(df)

        self.J_inv = np.linalg.pinv(J)

    def get_forces_and_moments(self, v_B_B_BA, v_inf, C_BW, u, w_B_B_BE):
        F = np.array([0, 0, 0])
        M = np.array([0, 0, 0])

        for i in range(len(self.surfs)):
            self.surfs[i].forces_and_moments(v_B_B_BA, v_inf, C_BW, u[i], w_B_B_BE)
            F = F + self.surfs[i].get_surface_forces()
            M = M + self.surfs[i].get_surface_moments()

        return F, M


# Start with a derivative controller, look at attitude/omega
# Best way to show benefit of controller is to start system out in trim
# Introduce a disturbance, like wind gust, tail wind is good, because it steels
# Airspeed, it'll take aircraft 60 seconds to reject the disturbance
# Objective is to decrease the settling time, and decrease the magnitude of the
# Deflections, all you have to do is single gain that calculates pitch rate Q
# if you are doing an ODE solver, for 150 seconds, elevator depends on pitch
# rate, and I want to implement a feed back control system, and I'm going
# To feedback from Q to my elevator, if the aircraft is pitching down a lot
# I want apply nose up elevator, if it's pitching  up, I want to apply some
# nose-down elevator
