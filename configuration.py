import os
import configparser
from re import I

config = configparser.ConfigParser()

# Create config folder if it does not exist
config_dir = "configs"
os.makedirs(config_dir, exist_ok=True)

maximums_quat = {
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

maximums_euler = {
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

columns_euler = list(maximums_euler.keys())

# Deep Neural Network Configuration
# Configuring DNN inputs
input_dict = {"Time": 10}
input_dict.update(maximums_quat)

[*input_columns], [*input_norm_factor] = zip(*input_dict.items())
input_dim = len(input_columns)

# Configuring DNN outputs
output_dict = maximums_quat
[*output_columns], [*output_norm_factor] = zip(*output_dict.items())
output_dim = len(output_columns)

path = os.path.join(config_dir, "DNN_default.ini")
config["DEFAULTS"] = {
    "INPUT_COLUMNS": input_columns,
    "INPUT_COLUMNS_EULER": ["Time"] + columns_euler,
    "INPUT_NORM_FACTORS": input_norm_factor,
    "INPUT_DIM": input_dim,
    "OUTPUT_COLUMNS": output_columns,
    "OUTPUT_COLUMNS_EULER": columns_euler,
    "OUTPUT_NORM_FACTORS": output_norm_factor,
    "OUTPUT_DIM": output_dim,
    "NETWORK_TYPE": "DNN",
}

config["DIR"] = {
    "model": "models",
    "checkpoints": "model_checkpoints",
    "tuner": "tuner",
    "training": "training_data",
    "test": "test_data",
}

config["NETWORK"] = dict(
    activation_functions=["tanh", "sigmoid"],
    depth_range=[6, 12, 2],
    width_range=[20, 50, 2],
    regularization_range=[1e-11, 0.001224, "log"],
    kernel_initializer_choices=[
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "he_uniform",
        "random_normal",
        "random_uniform",
        "zeros",
    ],
    bias_initializer_choices=["zeros", "random_uniform"],
    learning_rate_choices=[1e-2, 1e-3, 1e-4],
    optimizer="adam",
    metrics=["mse"],
)

with open(path, "w") as configfile:
    config.write(configfile)


# Recurrent Neural Network Configuration

# Configuring RNN inputs
input_dict = maximums_quat
[*input_columns], [*input_norm_factor] = zip(*input_dict.items())
input_dim = len(input_columns)

# Configuring RNN outputs
output_dict = maximums_quat
[*output_columns], [*output_norm_factor] = zip(*output_dict.items())
output_dim = len(output_columns)

path = os.path.join(config_dir, "RNN_default.ini")
config["DEFAULTS"] = {
    "INPUT_COLUMNS": input_columns,
    "INPUT_COLUMNS_EULER": columns_euler,
    "INPUT_NORM_FACTORS": input_norm_factor,
    "INPUT_DIM": input_dim,
    "OUTPUT_COLUMNS": output_columns,
    "OUTPUT_COLUMNS_EULER": columns_euler,
    "OUTPUT_NORM_FACTORS": output_norm_factor,
    "OUTPUT_DIM": output_dim,
    "NETWORK_TYPE": "RNN",
}

config["DIR"] = {
    "model": "models",
    "checkpoints": "model_checkpoints",
    "tuner": "tuner",
    "training": "training_data",
    "test": "test_data",
}

config["NETWORK"] = dict(
    activation_functions=["tanh", "sigmoid"],
    depth_range=[6, 12, 2],
    width_range=[20, 50, 2],
    regularization_range=[1e-11, 0.001224, "log"],
    kernel_initializer_choices=[
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "he_uniform",
        "random_normal",
        "random_uniform",
        "zeros",
    ],
    bias_initializer_choices=["zeros", "random_uniform"],
    learning_rate_choices=[1e-2, 1e-3, 1e-4],
    optimizer="adam",
    metrics=["mse"],
)

with open(path, "w") as configfile:
    config.write(configfile)
