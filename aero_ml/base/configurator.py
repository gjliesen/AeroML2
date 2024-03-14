import os
import typing
from typing import Union
import json
from pathlib import Path
from aero_ml.defaults import Defaults


class BaseConfigurator:
    config: dict  # dictionary containing the configuration

    def __init__(self, config_path: Union[os.PathLike, str], config_name: str = ""):
        """Configurator for the FF network, sets up default parameter config for network
        but the user can create new ones by calling the methods

        Args:
            config_dir (str): directory to store the configuration file
            config_name (str): name of the configuration file
        """
        self.path = Path(config_path)

        # If a full path is not provided the user must specify the config_name
        if self.path.is_dir() and config_name == "":
            raise ValueError(
                "config_name must be specified if config_dir is a directory"
            )
        elif self.path.is_dir():
            self.path = self.path / f"{config_name}.json"

        # Initialize the dictionary
        self.config = dict(
            config_name=self.path.stem,
            config_dir=str(self.path.parent),
        )

        # creating flags for if a function was run
        self.flags = dict(
            general_network=False,
            general_data=False,
            generation_data=False,
            tuning_data=False,
        )

    def general_network(
        self,
        input_dim: int = 14,
        output_dim: int = 13,
        optimizer: str = "adam",
        metrics: list[str] = ["mse"],
        loss_fn_str: str = "mse",
    ):
        self.flags["general_network"] = True
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
        self.flags["general_data"] = True
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
        iterations: int = 20000,
        test_cases: int = 30,
        constraints: dict = Defaults.constraints,
        rnd_method: str = "random",
        shuffle: bool = False,
    ):
        self.flags["generation_data"] = True
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
        self.flags["tuning_data"] = True
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

    def write(self, indent=1):
        with self.path.open("w") as outfile:
            json.dump(self.config, outfile, indent=indent)

    def generate(self):
        methods = [
            self.general_network,
            self.general_data,
            self.generation_data,
            self.tuning_data,
        ]
        for method, key in zip(methods, self.flags):
            if not self.flags[key]:
                # print(self.flags[key])
                # print(method.__name__)
                method()
        self.write()
