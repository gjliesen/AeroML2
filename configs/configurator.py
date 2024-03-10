import json
import os

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


class Configurator:
    def __init__(self, config_dir: str, config_name: str):
        self.config = dict(
            config_name=config_name,
            config_dir=config_dir,
        )
        self.fname = f"{config_dir}/{config_name}.json"

    def configure(self, network=None, dirs=None, tuner=None, data=None):
        if network is not None:
            self.set_network(network)
        if dirs is not None:
            self.set_dirs(dirs)
        if tuner is not None:
            self.set_tuner(tuner)
        if data is not None:
            self.set_data(data)

    def set_network(self, network):
        self.config.update(network)

    def set_dirs(self, dirs):
        self.config.update(dirs)

    def set_tuner(self, tuner):
        self.config.update(tuner)

    def set_data(self, data):
        self.config.update(data)

    def write(self, indent=1):
        with open(self.fname, "w") as outfile:
            json.dump(self.config, outfile, indent=indent)
