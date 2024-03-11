import time
import os
import json
import typing
from datetime import datetime
from aero_ml.defaults import Defaults


def timeit(f):
    """Decorator to time the execution of a function"""

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return timed


def parse_datetime(filename: str) -> datetime:
    """parses the datetime strings in a folder

    Args:
        filename (str):

    Returns:
        datetime: list of datetimestrings
    """
    date_time_str = filename.split("_")[:2]
    date_time_str = "_".join(date_time_str)
    return datetime.strptime(date_time_str, "%m%d%Y_%H%M%S")


def get_most_recent(folder_path: str) -> str:
    """Retrieves the most recent file from the specified directory

    Args:
        folder_path (str): path to the directory containing the file

    Raises:
        ValueError: folder_path must be a string
        FileNotFoundError: No recent file found in the specified directory

    Returns:
        str: path to the most recent file
    """
    # Ensure the folder path is a string
    if not isinstance(folder_path, str):
        raise ValueError("folder_path must be a string")

    # List all files in the directory
    files = os.listdir(folder_path)

    # Sort files by datetime
    sorted_files = sorted(files, key=parse_datetime, reverse=True)  # type: ignore

    # Return the most recent file
    if sorted_files:
        path = os.path.join(folder_path, sorted_files[0])
        return path
    else:
        raise FileNotFoundError("No recent file found in the specified directory")


class BaseConfigurator:
    config: dict  # dictionary containing the configuration

    def __init__(self, config_dir: str, config_name: str):
        """Configurator for the FF network, sets up default parameter config for network
        but the user can create new ones by calling the methods

        Args:
            config_dir (str): directory to store the configuration file
            config_name (str): name of the configuration file
        """
        self.config = dict(
            config_name=config_name,
            config_dir=config_dir,
        )
        self.fname = f"{config_dir}/{config_name}.json"

    def general_network(
        self,
        input_dim: int = 14,
        output_dim: int = 13,
        optimizer: str = "adam",
        metrics: list[str] = ["mse"],
        loss_fn_str: str = "mse",
    ):
        network = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            optimizer=optimizer,
            metrics=metrics,
            loss_fn_str=loss_fn_str,
        )

        self.config.update(network)

    def general_data(
        self,
        maximums_euler: dict = Defaults.MAXIMUMS_EULER,
        maximums_quat: dict = Defaults.MAXIMUMS_QUAT,
    ):
        features_euler = list(maximums_euler.keys())
        input_dict = {"time": 10}
        input_dict.update(maximums_quat)

        [*input_features], [*input_norm_factors] = zip(*input_dict.items())

        # Configuring DNN outputs
        output_dict = maximums_quat
        [*output_features], [*output_norm_factors] = zip(*output_dict.items())

        data = dict(
            input_features=input_features,
            input_features_euler=["Time"] + features_euler,
            input_norm_factors=input_norm_factors,
            output_features=output_features,
            output_features_euler=features_euler,
            output_norm_factors=output_norm_factors,
        )

        self.config.update(data)

    def generation_data(
        self,
        run_time: int = 10,
        frequency: float = 0.01,
        iterations: int = 5000,
        test_cases: int = 30,
        constraints: dict = Defaults.constraints,
        rnd_method: str = "random",
        shuffle: bool = False,
    ):
        data = dict(
            run_time=run_time,
            frequency=frequency,
            iterations=iterations,
            length=(run_time / frequency) * iterations,
            test_cases=test_cases,
            constraints=constraints,
            rnd_method=rnd_method,
            shuffle=shuffle,
        )

        self.config.update(data)

    def tuning_data(
        self,
        width_range: list[int] = [20, 50, 2],
        depth_range: list[int] = [6, 12, 2],
        activation_fns: list[str] = Defaults.activation_fns,
        reg_range: list[typing.Union[float, str]] = Defaults.reg_range,
        kernel_inits: list[str] = Defaults.kernel_inits,
        bias_inits: list[str] = Defaults.bias_inits,
        learning_rates: list[float] = Defaults.learning_rates,
    ):
        tuner = dict(
            activation_fns=activation_fns,
            width_range=width_range,
            depth_range=depth_range,
            reg_range=reg_range,
            kernel_inits=kernel_inits,
            bias_inits=bias_inits,
            learning_rates=learning_rates,
        )

        self.config.update(tuner)

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
