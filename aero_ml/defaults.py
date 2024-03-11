import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    MAXIMUMS_QUAT = {
        "Lat": 90,  # Degrees
        "Lon": 90,  # Degrees
        "Alt": 12000,  # Meters
        "q0": 1,  # component attitude quaternion
        "q1": 1,  # component attitude quaternion
        "q2": 1,  # component attitude quaternion
        "q3": 1,  # component attitude quaternion
        "Vx": 60,  # m/s
        "Vy": 20,  # m/s
        "Vz": 20,  # m/s
        "P": 12,  # Roll Rate deg/s
        "Q": 12,  # Pitch Rate deg/s
        "R": 12,  # Yaw Rate deg/s
    }

    MAXIMUMS_EULER = {
        "Lat": 90,  # Degrees
        "Lon": 90,  # Degrees
        "Alt": 12000,  # Meters
        "Phi": 90,  # attitude roll component degrees
        "Theta": 90,  # attitude theta component degrees
        "Psi": 90,  # attitdue yaw component degrees
        "Vx": 60,  # m/s
        "Vy": 20,  # m/s
        "Vz": 20,  # m/s
        "P": 12,  # Roll Rate deg/s
        "Q": 12,  # Pitch Rate deg/s
        "R": 12,  # Yaw Rate deg/s
    }

    constraints = {
        "lat": (-10, 10),
        "lon": (-10, 10),
        "alt": (300, 12000),
        "phi": ((-np.pi / 32), (np.pi / 32)),  # Radians?
        "theta": ((-np.pi / 16), (np.pi / 16)),  # Radians?
        "psi": ((-np.pi / 32), (np.pi / 32)),  # Radians?
        "v_x": (12.5, 60),  # m/s
        "v_y": (-10, 10),  # m/s
        "v_z": (-20, 20),  # m/s
        "P": (-10, 10),  # Rad / S
        "Q": (-10, 10),  # Rad / S
        "R": (-10, 10),  # Rad / S
    }

    activation_fns = ["relu", "tanh", "sigmoid"]

    reg_range = [1e-11, 0.001224, "log"]

    kernel_inits = [
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "he_uniform",
        "random_normal",
        "random_uniform",
        "zeros",
    ]

    bias_inits = ["zeros", "random_uniform"]

    learning_rates = [1e-2, 1e-3, 1e-4]
