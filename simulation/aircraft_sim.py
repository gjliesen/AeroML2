import numpy as np
import plotly.graph_objects as go
from numba import jit
from numpy import sin, cos, tan
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import pyproj

from .aero_parameters import Aero_Parameters

transformer = pyproj.Transformer.from_crs(
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
)


@jit
def quat_to_euler(quat):
    """
    Converts quaternion to euler angles and returns as numpy array
    :param quat: quaternion of the current aircraft attitude
    :return: array of the current aircraft attitude
    """
    psi_num = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
    psi_den = 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
    psi = np.arctan2(psi_num, psi_den)

    theta = np.arcsin(2 * (quat[0] * quat[2] - quat[1] * quat[3]))

    phi_num = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
    phi_den = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    phi = np.arctan2(phi_num, phi_den)

    return np.array([phi, theta, psi])


@jit
def quat_DCM(q):
    """
    This function builds a DCM, C_BN, from a quaternion
    :param q: quaternion of the current aircraft attitude
    :return: DCM of the aircraft built from the given quaternion
    """
    q0q1 = q[0] * q[1]
    q0q2 = q[0] * q[2]
    q0q3 = q[0] * q[3]
    q1q2 = q[1] * q[2]
    q1q3 = q[1] * q[3]
    q2q3 = q[2] * q[3]

    x11 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    x22 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    x33 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    x12 = 2 * (q1q2 + q0q3)
    x13 = 2 * (q1q3 - q0q2)
    x21 = 2 * (q1q2 - q0q3)
    x23 = 2 * (q2q3 + q0q1)
    x31 = 2 * (q1q3 + q0q2)
    x32 = 2 * (q2q3 - q0q1)

    C_BA = np.array([[x11, x12, x13],
                     [x21, x22, x23],
                     [x31, x32, x33]])

    return C_BA


@jit(nopython=True)
def matrix_vector(x, y):
    return x @ y.T


@jit(nopython=True)
def matrix_matrix(x, y):
    return x @ y


@jit
def euler_to_quat(phi, theta, psi):
    cphi_2 = cos(phi * 0.5)
    cpsi_2 = cos(psi * 0.5)
    ctheta_2 = cos(theta * 0.5)
    sphi_2 = sin(phi * 0.5)
    spsi_2 = sin(psi * 0.5)
    stheta_2 = sin(theta * 0.5)

    q_0 = (cphi_2 * ctheta_2 * cpsi_2) + (sphi_2 * stheta_2 * spsi_2)
    q_1 = (sphi_2 * ctheta_2 * cpsi_2) - (cphi_2 * stheta_2 * spsi_2)
    q_2 = (cphi_2 * stheta_2 * cpsi_2) + (sphi_2 * ctheta_2 * spsi_2)
    q_3 = (cphi_2 * ctheta_2 * spsi_2) - (sphi_2 * stheta_2 * cpsi_2)

    return np.array([q_0, q_1, q_2, q_3])


@jit
def euler_to_dcm(phi, theta, psi):
    # Taking in Euler angles to get DCM
    x_11 = cos(theta) * cos(psi)
    x_12 = cos(theta) * sin(psi)
    x_13 = -sin(theta)

    x_21 = -cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi)
    x_22 = cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi)
    x_23 = sin(phi) * cos(theta)

    x_31 = sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)
    x_32 = -sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)
    x_33 = cos(phi) * cos(theta)

    c_n_b = np.array([[x_11, x_12, x_13],
                      [x_21, x_22, x_23],
                      [x_31, x_32, x_33]]).T

    return c_n_b


@jit
def ecef_to_ned_matrix(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    x_11 = -cos(lon) * sin(lat)
    x_12 = -sin(lon)
    x_13 = -cos(lat) * cos(lon)
    x_21 = -sin(lat) * sin(lon)
    x_22 = cos(lon)
    x_23 = -cos(lat) * sin(lon)
    x_31 = cos(lat)
    x_32 = 0
    x_33 = -sin(lat)

    C_EN = np.array([[x_11, x_12, x_13],
                     [x_21, x_22, x_23],
                     [x_31, x_32, x_33]])

    return C_EN


@jit
def get_w_B_B_BE_matrix(P, Q, R):
    w_B_B_BE_matrix = np.array([[0, -P, -Q, -R],
                                [P, 0, R, -Q],
                                [Q, -R, 0, P],
                                [R, Q, -P, 0]])
    return w_B_B_BE_matrix


@jit
def cross_product_matrix(vec):
    cpm = np.array([[0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]])
    return cpm


@jit
def body_to_wind_matrix(alpha, beta):
    sacb = sin(alpha) * cos(beta)
    cacb = cos(alpha) * cos(beta)
    casb = cos(alpha) * sin(beta)
    sasb = sin(alpha) * sin(beta)

    C_BW = np.array([[cacb, -casb, -sin(alpha)],
                     [sin(beta), cos(beta), 0],
                     [sacb, -sasb, cos(alpha)]])

    return C_BW


@jit
def h_theta_matrix(phi, theta):
    x_11 = 1
    x_12 = tan(theta) * sin(phi)
    x_13 = tan(theta) * cos(phi)
    x_21 = 0
    x_22 = cos(phi)
    x_23 = -sin(phi)
    x_31 = 0
    x_32 = sin(phi) / cos(theta)
    x_33 = cos(phi) / cos(theta)

    h_theta = np.array([[x_11, x_12, x_13],
                        [x_21, x_22, x_23],
                        [x_31, x_32, x_33]])

    return h_theta


# noinspection DuplicatedCode
class Aircraft_Sim:
    # noinspection SpellCheckingInspection
    def __init__(self, x_E_E_BE_0, att_B_B_BN_0, v_E_B_BE_0, w_B_B_BE_0,
                 Fext_B_0, Mext_B_0, g_n):
        # noinspection SpellCheckingInspection
        """
                takes initial aircraft states and runs 6DOF simulation

                :param x_E_E_BE_0: initial position of aircraft
                :param att_B_B_BN_0: initial attitude of aircraft
                :param v_E_B_BE_0: initial velocity of aircraft
                :param w_B_B_BE_0: initial body rates of aircraft
                :param Fext_B_0: initial external forces of aircraft
                :param Mext_B_0: initial external moments of aircraft
                :param g_n: gravity
                """
        # Initial Values
        phi, theta, psi = att_B_B_BN_0
        attQuat_B_B_BN_0 = euler_to_quat(phi, theta, psi)

        self.states_init = np.hstack((x_E_E_BE_0, attQuat_B_B_BN_0,
                                      v_E_B_BE_0, w_B_B_BE_0))
        self.init_step = True
        self.tk_flag = True
        self.lla_flag = True
        self.ae_flag = True

        # Trim conditions
        self.theta_trim = -0.08385007
        self.delta_e_trim = -0.01884321

        # Current States
        self.x_E_E_BE = None
        self.att_B_B_BN = None
        self.attQuat_B_B_BN = None
        self.v_E_B_BE = None
        self.w_B_B_BE = None

        self.states_cur = None  # All states stored in one 1xN array
        # Current DCMs
        self.C_NB = None
        self.C_BN = None
        self.C_BW = None
        self.C_EN = None

        # Gains
        self.k_d_pitch = None
        self.k_p_pitch = None
        self.k_d_roll = None
        self.k_p_roll = None

        # Constants
        self.g_n = np.array([0, 0, g_n])
        self.Fext_B = Fext_B_0
        self.Mext_B = Mext_B_0
        self.aircraft = Aero_Parameters()

        # Aerodynamics
        self.v_inf = None
        self.v_b_a = None
        self.v_a_n_gust = None
        self.v_a_n = np.array([0, 0, 0])
        self.alpha = None
        self.beta = None
        self.solver_time = []
        # Stored States

        # Main States
        self.x_E_E_BE_states = x_E_E_BE_0
        self.attQuat_B_B_BN_states = attQuat_B_B_BN_0
        self.v_E_B_BE_states = v_E_B_BE_0
        self.w_B_B_BE_states = w_B_B_BE_0

        # Secondary States
        self.att_B_B_BN_states = []
        self.p_E_E_BE_states = []
        self.w_B_BA_states = []

        self.time = None

    def aero_inputs(self):
        """
        Calculates the V_inf, AOA, and side slip of aircraft at each time step
        :return:
        """
        self.v_b_a = self.v_E_B_BE - \
                     matrix_vector(self.C_BN, self.v_a_n.astype(np.float64))

        self.v_inf = np.linalg.norm(self.v_b_a)
        if np.isnan(self.v_inf):
            self.v_inf = 0
            self.beta = 0
            self.alpha = 0
        else:
            self.alpha = np.arctan2(self.v_b_a[2], self.v_b_a[0])
            self.beta = np.arcsin(self.v_b_a[1] / self.v_inf)

        w_B_BA = np.array([self.v_inf,
                           self.alpha * (180 / np.pi),
                           self.beta * (180 / np.pi)])
        self.w_B_BA_states.append(w_B_BA)

    def translational_kinematics(self):
        """
        This function takes the current position vector, x_E_E_BE, velocity
        vector, v_E_E_BE, and a DCM, C_NB, to find the change in position of
        the aircraft over time.
        :return: xDot_E_E_BE
        The return variable is later integrated to get the next
        position state of the aircraft
        """

        # We currently have the aircraft position in cartesian coordinates
        # we need to convert this to Polar to get the new DCM.

        lat, lon, alt = transformer.transform(self.x_E_E_BE[0],
                                              self.x_E_E_BE[1],
                                              self.x_E_E_BE[2],
                                              radians=False)

        # lat, lon, alt = navpy.ecef2lla(self.x_E_E_BE, 'deg')
        p_E_E_BE = np.array([lat, lon, alt])

        self.p_E_E_BE_states.append(p_E_E_BE)
        self.C_EN = ecef_to_ned_matrix(lat, lon)

        # The C_EB matrix to get the velocity given in the Body coordinate
        # system to the earth coordinate system
        C_EB = self.C_EN @ self.C_NB

        xDot_E_E_BE = matrix_vector(C_EB, self.v_E_B_BE)

        return xDot_E_E_BE

    def rotational_kinematics_quat(self):
        """
        This function takes the current attitude quaternion, attQuat_B_B_BN,
        and body rates, w_B_B_BE to calculate the change in the attitude
        quaternion over time, attQuatDot_B_B_BN.
        :return: attQuatDot_B_B_BN
        The return variable is later integrated to get the next attitude state
        of the aircraft in quaternion form
        """

        P, Q, R = self.w_B_B_BE
        w_B_B_BE_matrix = get_w_B_B_BE_matrix(P, Q, R)

        attQuatDot_B_B_BN = 0.5 * (w_B_B_BE_matrix @ self.attQuat_B_B_BN.T)

        return attQuatDot_B_B_BN

    def rotational_kinematics_euler(self):
        phi, theta, psi = self.att_B_B_BN

        h_theta = h_theta_matrix(phi, theta)

        attDot_B_B_BN = matrix_vector(h_theta, self.w_B_B_BE)

        return attDot_B_B_BN

    def translational_dynamics(self):
        """
        This function takes in the current body rates, w_B_B_BE, velocity,
        v_E_E_BE, and Forces, Fext_B, to calculate the change in velocity over
        time, vDot_E_B_BE.
        :return: vDot_E_E_BE
        The return variable is later integrated to get the next velocity state
        of the aircraft.
        """

        w_B_B_BE_mat = cross_product_matrix(self.w_B_B_BE)

        vDot_E_B_BE = (1 / self.aircraft.mass) * self.Fext_B + \
                      matrix_vector(self.C_BN, self.g_n) - \
                      matrix_vector(w_B_B_BE_mat, self.v_E_B_BE)

        return vDot_E_B_BE

    def rotational_dynamics(self):
        """
        This function takes in the current body rates, w_B_B_BE, and the
        current moments, Mext_B, to calculate the change in body rates over
        time, wDot_B_B_BE
        :return: wDot_B_B_BE
        the return variable is later integrated to get the next body rate state
        of the aircraft.
        """
        w_B_B_BE_mat = cross_product_matrix(self.w_B_B_BE)

        temp = matrix_vector(w_B_B_BE_mat,
                             (matrix_vector(self.aircraft.J, self.w_B_B_BE)))

        wDot_B_B_BE = matrix_vector(self.aircraft.J_inv, (self.Mext_B - temp))

        return wDot_B_B_BE

    def aircraft_EOM(self, t, y):
        """
        Takes in an initial state and returns the derivative of that state to
        the ODE solver, this function also updates the state member variables,
        aircraft aerodynamics, and DCMs.

        :param t: Current time step
        :param y: Current state of aircraft
        :return: Output state to be integrated and fed back into the solver
        """
        # Update Step
        # States Update:
        self.x_E_E_BE = y[0:3]
        self.attQuat_B_B_BN = y[3:7]
        self.v_E_B_BE = y[7:10]
        self.w_B_B_BE = y[10:13]

        self.attQuat_B_B_BN = self.attQuat_B_B_BN / np.linalg.norm(
            self.attQuat_B_B_BN)

        # DCM Update:
        self.C_BN = quat_DCM(self.attQuat_B_B_BN)
        self.C_NB = self.C_BN.T

        # Aerodynamics:
        self.solver_time.append(t)
        if t >= 20:
            self.v_a_n = self.v_a_n_gust

        self.aero_inputs()
        self.C_BW = body_to_wind_matrix(self.alpha, self.beta)
        # Pitch Damper
        att_B_B_BN = quat_to_euler(self.attQuat_B_B_BN)

        delta_e = self.k_d_roll * self.w_B_B_BE[1] + \
                  self.k_p_pitch * (att_B_B_BN[1] - self.theta_trim) + \
                  self.delta_e_trim

        # Roll Damper
        delta_R = self.k_d_roll * self.w_B_B_BE[0] + \
                  self.k_p_roll * att_B_B_BN[0]

        delta_L = -delta_R

        deflections = np.array([delta_R, delta_L, delta_e, 0])

        self.Fext_B, self.Mext_B = \
            self.aircraft.get_forces_and_moments(self.v_b_a,
                                                 self.v_inf,
                                                 self.C_BW,
                                                 deflections,
                                                 self.w_B_B_BE)

        # Equations of Motion:
        xDot_E_E_BE = self.translational_kinematics()
        attQuatDot_B_B_BN = self.rotational_kinematics_quat()
        vDot_E_B_BE = self.translational_dynamics()
        wDot_B_B_BE = self.rotational_dynamics()

        # Final State Matrix
        output_states = np.hstack((xDot_E_E_BE, attQuatDot_B_B_BN,
                                   vDot_E_B_BE, wDot_B_B_BE))
        return output_states

    def run_simulation(self, time, k_d_pitch, k_p_pitch, k_d_roll,
                       k_p_roll, v_a_n):
        # Setting Control Gains
        self.k_d_pitch = k_d_pitch
        self.k_p_pitch = k_p_pitch
        self.k_d_roll = k_d_roll
        self.k_p_roll = k_p_roll
        self.v_a_n_gust = v_a_n

        # Defining time array
        self.time = np.arange(0, time, 0.01)

        result = solve_ivp(self.aircraft_EOM, t_span=[0, time],
                           t_eval=self.time, y0=self.states_init)

        state_sol = result.y.T
        self.x_E_E_BE_states = state_sol[0::, 0:3]
        self.attQuat_B_B_BN_states = state_sol[0::, 3:7]
        for states in self.attQuat_B_B_BN_states:
            row = quat_to_euler(states)
            self.att_B_B_BN_states.append(row)

        self.att_B_B_BN_states = np.array(self.att_B_B_BN_states)
        self.w_B_BA_states = np.array(self.w_B_BA_states)
        self.p_E_E_BE_states = np.array(self.p_E_E_BE_states)
        self.v_E_B_BE_states = state_sol[0::, 7:10]
        self.w_B_B_BE_states = state_sol[0::, 10:13]

    def state_graph(self, state, test_case):
        """
        :param test_case:
        :param state: desired state to plot individually
        :return:
        """
        state_dict = {
            "x_E_E_BE": self.x_E_E_BE_states,
            "att_B_B_BN": self.att_B_B_BN_states,
            "v_E_B_BE": self.v_E_B_BE_states,
            "w_B_B_BE": self.w_B_B_BE_states,
            "p_E_E_BE": self.p_E_E_BE_states,
            "w_B_BA": self.w_B_BA_states
        }

        x_dict = {
            "x_E_E_BE": "X",
            "att_B_B_BN": "Phi",
            "v_E_B_BE": "V_x",
            "w_B_B_BE": "P",
            "p_E_E_BE": "Lat",
            "w_B_BA": "V_inf"
        }

        y_dict = {
            "x_E_E_BE": "Y",
            "att_B_B_BN": "Theta",
            "v_E_B_BE": "V_y",
            "w_B_B_BE": "Q",
            "p_E_E_BE": "Lon",
            "w_B_BA": "Alpha"
        }

        z_dict = {
            "x_E_E_BE": "Z",
            "att_B_B_BN": "Psi",
            "v_E_B_BE": "V_z",
            "w_B_B_BE": "R",
            "p_E_E_BE": "Alt",
            "w_B_BA": "Beta"
        }

        states = state_dict[state]

        fig = go.Figure()

        # if state == "x_E_E_BE":
        #     states[0::, 2] = states[0::, 2] - 6378e3
        if state == "w_B_BA" or state == "p_E_E_BE":
            time = self.solver_time
        else:
            time = self.time
        fig.add_trace(go.Scattergl(x=time, y=states[0::, 0],
                                   name=x_dict[state]))

        fig.add_trace(go.Scattergl(x=time, y=states[0::, 1],
                                   name=y_dict[state]))

        fig.add_trace(go.Scattergl(x=time, y=states[0::, 2],
                                   name=z_dict[state]))

        html_title = state + '_' + test_case + '.html'
        fig.write_html(html_title)
        fig.show()

    def all_states_graph(self, title):
        """
        Plots all states with a shared x-axis
        :return:
        """
        state_dict = {
            "x_E_E_BE": self.x_E_E_BE_states,
            "att_B_B_BN": self.att_B_B_BN_states,
            "v_E_B_BE": self.v_E_B_BE_states,
            "w_B_B_BE": self.w_B_B_BE_states,
            "p_E_E_BE": self.p_E_E_BE_states,
            "w_B_BA": self.w_B_BA_states
        }

        x_dict = {
            "x_E_E_BE": "X",
            "att_B_B_BN": "Phi",
            "v_E_B_BE": "V_x",
            "w_B_B_BE": "P",
            "p_E_E_BE": "Lat",
            "w_B_BA": "V_inf"
        }

        y_dict = {
            "x_E_E_BE": "Y",
            "att_B_B_BN": "Theta",
            "v_E_B_BE": "V_y",
            "w_B_B_BE": "Q",
            "p_E_E_BE": "Lon",
            "w_B_BA": "Alpha"
        }

        z_dict = {
            "x_E_E_BE": "Z",
            "att_B_B_BN": "Psi",
            "v_E_B_BE": "V_z",
            "w_B_B_BE": "R",
            "p_E_E_BE": "Alt",
            "w_B_BA": "Beta"
        }

        state_list = ["x_E_E_BE", "att_B_B_BN", "v_E_B_BE", "w_B_B_BE"]

        fig = make_subplots(rows=4, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02,
                            subplot_titles=state_list)

        i = 1

        for state in state_list:
            states = state_dict[state]

            if state == "w_B_BA" or state == "p_E_E_BE":
                time = self.solver_time
                print("entered")
            else:
                time = self.time

            fig.add_trace(go.Scattergl(x=time, y=states[0::, 0],
                                       name=x_dict[state]),
                          row=i, col=1)

            fig.add_trace(go.Scattergl(x=time, y=states[0::, 1],
                                       name=y_dict[state]),
                          row=i, col=1)

            fig.add_trace(go.Scattergl(x=time, y=states[0::, 2],
                                       name=z_dict[state]),
                          row=i, col=1)

            i += 1

        fig.update_layout(hovermode='x unified', title=title,
                          height=1000)
        fig.show()
        # fig.write_image("C:/Users/gjlie/PycharmProjects/AME_532a_Optimized"
        #                 "/Plots/" +
        #                 title + ".jpg")

