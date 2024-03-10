import numpy as np
from numba import jit
from numba.experimental import jitclass
from .jit_specs import surface_two_spec1d

# import logging


zeros = np.zeros((3, 3))


@jit(nopython=True)
def cross_product_matrix(vec):
    cpm = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    return cpm


@jit(nopython=True)
def matrix_vector(x, y):
    return x @ y.T


@jit(nopython=True)
def get_c_moment(M_surf_W):
    c_moment = np.array([[0, 0, 0], [0, 0, -1 * M_surf_W], [0, 1 * M_surf_W, 0]])
    return c_moment


@jit(nopython=True)
def eye_mat(mat1, mat2, mat3, mat4):
    row1 = np.hstack((mat1, zeros, zeros, zeros))
    row2 = np.hstack((zeros, mat2, zeros, zeros))
    row3 = np.hstack((zeros, zeros, mat3, zeros))
    row4 = np.hstack((zeros, zeros, zeros, mat4))

    return np.vstack((row1, row2, row3, row4))


# # Enable DEBUG level logging for Numba
# logging.basicConfig(level=logging.DEBUG)


@jitclass(surface_two_spec1d)
class AeroSurface:
    def __init__(self, arr, n_vec, dx_vec, dx_mat):
        self.surf_arr = arr

        self.chord = np.ascontiguousarray(self.surf_arr[:, 0])
        self.span = np.ascontiguousarray(self.surf_arr[:, 1])
        self.CL0 = np.ascontiguousarray(self.surf_arr[:, 2])
        self.e = np.ascontiguousarray(self.surf_arr[:, 3])
        self.i = np.ascontiguousarray(self.surf_arr[:, 4])
        self.CD0 = np.ascontiguousarray(self.surf_arr[:, 5])
        self.CDa = np.ascontiguousarray(self.surf_arr[:, 6])
        self.a0 = np.ascontiguousarray(self.surf_arr[:, 7])
        self.CM0 = np.ascontiguousarray(self.surf_arr[:, 8])
        self.CMa = np.ascontiguousarray(self.surf_arr[:, 9])

        self.n_vec = n_vec
        self.dx_vec = dx_vec
        self.dx_mat = dx_mat

        self.expansion_matrix = np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            np.float64,
        )

        # not contiguous
        self.contraction_matrix = np.ascontiguousarray(self.expansion_matrix.T)

        self.first_element_matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            np.float64,
        )
        eye_mat = np.eye(3).astype(np.float64)
        self.sum_matrix = np.vstack((eye_mat, eye_mat, eye_mat, eye_mat))

        self.n_drag_vector = np.array(
            [-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0], np.float64
        )

        self.dCLdu = 3.0

        self.w_B_B_BE_mat = np.zeros((12, 12))
        self.C_BW = np.zeros((12, 12))
        self.v_B_B_BA_vec = np.zeros((12,))
        self.u = np.zeros((4,))
        self.Sref = np.zeros((4,))
        self.AR = np.zeros((4,))
        self.CL_a = np.zeros((4,))
        self.Cu = np.zeros((4,))
        self.Su = np.zeros((4,))
        self.SuS = np.zeros((4,))
        self.a_surf = np.zeros((4,))
        self.q_bar = -1.0
        self.CL = np.zeros((4,))
        self.CD = np.zeros((4,))
        self.CM = np.zeros((4,))

        self.F_surf_B_vec = np.zeros((12,))
        self.L_surf_W = np.zeros((4,))
        self.M_surf_W = np.zeros((4,))
        self.D_surf_W = np.zeros((4,))

        self.c_moment_mat = np.zeros((12, 12))
        self.calc_parameters()

        # Intermediate parameters
        self.v_B_B_SnA = np.zeros((12,))
        self.v_temp = np.zeros((12,))

    def calc_parameters(self):
        C_Moment = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)

        # First instance of eye_mat
        self.c_moment_mat = eye_mat(C_Moment, C_Moment, C_Moment, C_Moment)

        self.Sref[:] = self.chord * self.span
        self.AR[:] = self.span**2 / self.Sref
        self.CL_a[:] = 2 * np.pi * self.AR / (2 + self.AR)
        self.Cu[:] = 0.2 * self.chord
        self.Su[:] = self.Cu * self.span
        self.SuS[:] = self.Su / self.Sref

    def forces_and_moments(self, q_bar, C_BW, u, v_B_B_BA, w_B_B_BE_mat):
        # Second instance of eye_mat
        for i in range(0, 12, 3):
            self.w_B_B_BE_mat[i : i + 3, i : i + 3] = w_B_B_BE_mat
            self.C_BW[i : i + 3, i : i + 3] = C_BW
            self.v_B_B_BA_vec[i : i + 3] = v_B_B_BA

        self.q_bar = q_bar
        self.u[:] = u

        self.calc_surface_AOA()
        F = self.calc_forces()
        M = self.calc_moments()

        return F, M

    def calc_surface_AOA(self):
        self.v_B_B_SnA[:] = self.v_B_B_BA_vec + np.dot(self.w_B_B_BE_mat, self.dx_vec)

        self.v_temp[:] = self.v_B_B_SnA * self.n_vec

        temp3 = np.dot(self.contraction_matrix, self.v_temp)
        temp4 = np.dot(self.first_element_matrix, self.v_B_B_SnA)

        self.a_surf[:] = self.i - np.arctan2(temp3, temp4)

    def calc_forces(self):
        self.calc_lift()
        self.calc_drag()
        F_surf_W = (
            np.dot(self.expansion_matrix, self.D_surf_W) * self.n_drag_vector
            + np.dot(self.expansion_matrix, self.L_surf_W) * self.n_vec
        )

        self.F_surf_B_vec[:] = np.dot(self.C_BW, F_surf_W)
        F_surf_B = np.dot(self.F_surf_B_vec, self.sum_matrix)
        return F_surf_B

    def calc_moments(self):
        self.CM[:] = self.CM0 + self.CMa * self.a_surf

        self.M_surf_W[:] = self.CM * self.q_bar * self.Sref * self.chord
        C_moment = np.dot(self.expansion_matrix, self.M_surf_W) * self.c_moment_mat
        M_surf_B_vec = np.dot(C_moment, self.n_vec) + np.dot(
            self.dx_mat, self.F_surf_B_vec
        )
        M_surf_B = np.dot(M_surf_B_vec, self.sum_matrix)
        return M_surf_B

    def calc_lift(self):
        # Calculating lift coefficient
        self.CL[:] = self.CL0 + self.CL_a * self.a_surf + self.SuS * self.dCLdu * self.u
        # Calculating lift force
        self.L_surf_W[:] = self.CL * self.q_bar * self.Sref

    def calc_drag(self):
        # Calculating drag coefficient
        a_diff = self.a_surf - self.a0
        self.CD[:] = (
            self.CD0 + self.CDa * a_diff**2 + self.CL**2 / (np.pi * self.AR) / self.e
        )
        # Calculating drag force
        self.D_surf_W[:] = self.CD * self.q_bar * self.Sref


class AeroParameters:
    def __init__(self):
        self.mass_props = np.load("aero_ml/simulation/mass_props.npy")
        self.surfs = None
        self.x_cm = None
        self.J = None
        self.J_inv = None
        self.mass = None
        self.dCLdu = 3.0
        self.rho = 1.225

        self.surf_df = None

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

        # S2 = Elevator = Surface 2
        s2_props = mass_props_ac[2, 0:]
        c_s2 = s2_props[1]
        b_s2 = max(s2_props[2], s2_props[3])

        x_s2 = np.array([s2_props[4] + (1 / 4) * s2_props[1], s2_props[5], s2_props[6]])

        dx_cm2 = x_s2 - self.x_cm
        dx_mat2 = cross_product_matrix(dx_cm2)

        # X_cm_surf, c, b, n
        # S3 = Rudder = Surface 3
        dx_cm3 = x_s3 - self.x_cm
        dx_mat3 = cross_product_matrix(dx_cm3)

        # S4 = Right Wing = Surface 4
        s4_props = mass_props_ac[0, 0:]
        c_s4 = s4_props[1]
        b_s4 = max(s4_props[2], s4_props[3])

        x_s4 = np.array([s4_props[4] + (1 / 4) * s4_props[1], s4_props[5], s4_props[6]])

        dx_cm4 = x_s4 - self.x_cm
        dx_mat4 = cross_product_matrix(dx_cm4)

        # S5 = Left Wing = Surface 5
        s5_props = mass_props_ac[1, 0:]
        c_s5 = s5_props[1]
        b_s5 = max(s5_props[2], s5_props[3])

        x_s5 = np.array([s5_props[4] + (1 / 4) * s5_props[1], s5_props[5], s5_props[6]])
        dx_cm5 = x_s5 - self.x_cm
        dx_mat5 = cross_product_matrix(dx_cm5)

        surf_list = [
            [c_s4, b_s4, CL0_s4, e_s4, i_s4, CD0_s4, CDa_s4, a0_s4, CM0_s4, CMa_s4],
            [c_s5, b_s5, CL0_s5, e_s5, i_s5, CD0_s5, CDa_s5, a0_s5, CM0_s5, CMa_s5],
            [c_s2, b_s2, CL0_s2, e_s2, i_s2, CD0_s2, CDa_s2, a0_s2, CM0_s2, CMa_s2],
            [c_s3, b_s3, CL0_s3, e_s3, i_s3, CD0_s3, CDa_s3, a0_s3, CM0_s3, CMa_s3],
        ]

        surf_arr = np.array(surf_list)

        n_vec = np.hstack((n_s4, n_s5, n_s2, n_s3), dtype=np.float64)
        dx_cm_vec = np.hstack((dx_cm4, dx_cm5, dx_cm2, dx_cm3), dtype=np.float64)
        # 4th instance of eye_mat
        dx_cm_mat = eye_mat(dx_mat4, dx_mat5, dx_mat2, dx_mat3)

        self.surfs2 = AeroSurface(surf_arr, n_vec, dx_cm_vec, dx_cm_mat)

    def mass_properties(self):
        self.get_mass()
        self.get_center_of_mass()
        self.get_mass_moi()

    def get_mass(self):
        # Convert to kg
        self.mass_props[0:, 0] = self.mass_props[0:, 0] / 1000
        self.mass = np.sum(self.mass_props[0:, 0])

    def get_center_of_mass(self):
        # Splitting calculations up into 3 separate matrices
        x_moments = self.mass_props[0:, 0] * self.mass_props[0:, 4]
        y_moments = self.mass_props[0:, 0] * self.mass_props[0:, 5]
        z_moments = self.mass_props[0:, 0] * self.mass_props[0:, 6]

        # Combining arrays into one moment matrix
        componentMoments = np.vstack((x_moments, y_moments, z_moments)).T

        # Summing columns and dividing by mass
        self.x_cm = (sum(componentMoments) / self.mass).T

    def get_mass_moi(self):
        inertias = np.empty((10, 3))
        for i in range(len(self.mass_props[0:, 0])):
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
        J[0, 0] = np.sum(inertias[0:, 0]) + np.sum(
            self.mass_props[0:, 0]
            * (
                np.square(self.mass_props[0:, 5] - self.x_cm[1])
                + np.square(self.mass_props[0:, 6] - self.x_cm[2])
            )
        )
        J[1, 1] = np.sum(inertias[0:, 1]) + np.sum(
            self.mass_props[0:, 0]
            * (
                np.square(self.mass_props[0:, 6] - self.x_cm[2])
                + np.square(self.mass_props[0:, 4] - self.x_cm[0])
            )
        )
        J[2, 2] = np.sum(inertias[0:, 2]) + np.sum(
            self.mass_props[0:, 0]
            * (
                np.square(self.mass_props[0:, 4] - self.x_cm[0])
                + np.square(self.mass_props[0:, 5] - self.x_cm[1])
            )
        )
        J[0, 1] = -np.sum(
            self.mass_props[0:, 0] * (self.mass_props[0:, 4] * self.mass_props[0:, 5])
        )
        J[0, 2] = -np.sum(
            self.mass_props[0:, 0] * (self.mass_props[0:, 6] * self.mass_props[0:, 4])
        )
        J[1, 2] = -np.sum(
            self.mass_props[0:, 0] * (self.mass_props[0:, 5] * self.mass_props[0:, 6])
        )

        J[0, 1] = -1 * J[0, 1]
        J[0, 2] = -1 * J[0, 2]
        J[1, 2] = -1 * J[1, 2]
        J[1, 0] = J[0, 1]
        J[2, 0] = J[0, 2]
        J[2, 1] = J[1, 2]

        self.J = J
        self.J_inv = np.linalg.pinv(J)

    def get_forces_and_moments(self, v_B_B_BA, v_inf, C_BW, u, w_B_B_BE):
        q_bar = 0.5 * self.rho * v_inf**2
        w_B_B_BE_mat = cross_product_matrix(w_B_B_BE)

        Ft, Mt = self.surfs2.forces_and_moments(q_bar, C_BW, u, v_B_B_BA, w_B_B_BE_mat)

        return Ft, Mt


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
