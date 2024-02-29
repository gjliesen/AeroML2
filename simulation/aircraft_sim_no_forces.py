import numpy as np
import navpy
import copy
from .aircraft_sim import Aircraft_Sim


class AircraftSimNoForces(Aircraft_Sim):
    def __init__(self, initial_conditions):
        ic_copy = copy.deepcopy(initial_conditions)
        lat, lon, alt = ic_copy[0:3]
        self.p_E_E_BE_0 = ic_copy[0:3]
        x_E_E_BE_0 = navpy.lla2ecef(lat, lon, alt)
        att_B_B_BN_0 = ic_copy[3:6]
        v_E_B_BE_0 = ic_copy[6:9]
        w_B_B_BE_0 = ic_copy[9:12]
        Fext_B_0 = np.array([0, 0, 0])
        Mext_B_0 = np.array([0, 0, 0])
        g_n = 9.81
        super().__init__(
            x_E_E_BE_0, att_B_B_BN_0, v_E_B_BE_0, w_B_B_BE_0, Fext_B_0, Mext_B_0, g_n
        )
        self.lla_states_init = np.hstack(
            (self.p_E_E_BE_0, self.states_init[3:7], v_E_B_BE_0, w_B_B_BE_0)
        )

    def get_all_states(self):
        x_E_E_BE = self.x_E_E_BE_states
        lat, lon, alt = navpy.ecef2lla(x_E_E_BE, latlon_unit="deg")
        p_E_E_BE = np.hstack(
            (lat.reshape(-1, 1), lon.reshape(-1, 1), alt.reshape(-1, 1))
        )
        # Output states need to be quaternions
        att_B_B_BN = self.attQuat_B_B_BN_states
        v_E_B_BE = self.v_E_B_BE_states
        w_B_B_BA = self.w_B_B_BE_states
        state_vec = np.hstack((p_E_E_BE, att_B_B_BN, v_E_B_BE, w_B_B_BA))
        ics = self.lla_states_init.reshape(1, 13)
        state_vec = np.concatenate((ics, state_vec), axis=0)

        return state_vec
