import numpy as np
from sim_data_engine import SimDataEngine

# Data Generation, Does not need to be ran every time
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

rate = 0.01  # 1/100 of second
total_time = 5  # seconds
size = 20000
# size = 1
title = "rnn_data"
engine = SimDataEngine()
engine.create_training_set(constraints, rate, total_time, size, "random", title, False)

# # Generate Test Cases
# for i in range(30):
#     rate = 0.01  # 1/100 of second
#     total_time = 10  # seconds
#     size = 1
#     title = f"unit{i}"
#     engine = SimDataEngine()
#     engine.create_training_set(
#         constraints, rate, total_time, size, "random", title, False
#     )
