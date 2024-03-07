from numba import float64, int64, int32, boolean

surface_two_spec = {
    "surf_arr": float64[:, :],  # 2D array of floats
    "n_vec": float64[:, :],  # 2D array of 32bit integers
    "dx_vec": float64[:, :],  # 2D array of floats
    "dx_mat": float64[:, :],  # 2D array of floats
    "contraction_matrix": float64[:, :],  # 2D array of 64 bit integers
    "expansion_matrix": float64[:, :],  # 2D array of 64 bit integers
    "first_element_matrix": float64[:, :],  # 2D array of 64 bit integers
    "sum_matrix": float64[:, :],  # 2D array of floats
    "n_drag_vector": float64[:, :],  # 2D array of 64 bit integers
    "dCLdu": float64,  # Single float value
    "w_B_B_BE_mat": float64[:, :],  # 2D array of floats
    "C_BW": float64[:, :],  # 2D array of floats
    "v_B_B_BA_vec": float64[:, :],  # 2D array of floats
    "u": float64[:],  # 1D array of float
    "Sref": float64[:],  # 1D array of float
    "AR": float64[:],  # 1D array of float
    "CL_a": float64[:],  # 1D array of float
    "Cu": float64[:],  # 1D array of float
    "Su": float64[:],  # 1D array of float
    "SuS": float64[:],  # 1D array of float
    "a_surf": float64[:, :],  # 2D array of float
    "q_bar": float64,  # Assuming single float
    "CL": float64,  # Assuming single float
    "CD": float64,  # Assuming single float
    "CM": float64,  # Assuming single float
    "F_surf_B_vec": float64[:, :],  # 2D array of floats
    "L_surf_W": float64[:],  # 1D array of floats
    "M_surf_W": float64[:],  # 1D array of floats
    "D_surf_W": float64[:],  # 1D array of floats
    "c_moment_mat": float64[:, :],  # 2D array of floats
    "v_B_B_SnA": float64[:, :],  # 2D array of floats
    "v_temp": float64[:, :],  # 2D array of floats
}

surface_spec = {
    "number": int64,
}

aero_parameters = {}
# <string>:3: NumbaPerformanceWarning: '@' is faster on contiguous arrays, called on (Array(float64, 2, 'A', False, aligned=True), Array(float64, 2, 'C', False, aligned=True))
# <string>:3: NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(float64, 1, 'A', False, aligned=True), Array(float64, 2, 'A', False, aligned=True))
# <string>:3: NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(float64, 2, 'A', False, aligned=True), Array(float64, 2, 'A', False, aligned=True))
# <string>:3: NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(float64, 2, 'A', False, aligned=True), Array(float64, 2, 'C', False, aligned=True))
# <string>:3: NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(float64, 2, 'C', False, aligned=True), Array(float64, 2, 'A', False, aligned=True))
# <string>:3: NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(float64, 2, 'F', False, aligned=True), Array(float64, 2, 'A', False, aligned=True))
