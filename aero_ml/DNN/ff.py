import typing
import os
import tensorflow as tf
from tensorflow import keras
from aero_ml.defaults import Defaults
from aero_ml.utils import BaseConfigurator
from aero_ml.base_data_engine import BaseDataEngine
from aero_ml.base_network_engine import BaseNetworkEngine
from aero_ml.base_test_engine import BaseTestEngine


class DataEngine(BaseDataEngine):
    def __init__(self, config: dict):
        super().__init__(config)

    def write_example(self, tfrecord: tf.io.TFRecordWriter):
        """Write example to tfrecord file

        FF uses lists as inputs and outputs so the inputs and outputs are just
        Args:
            tfrecord (typing.IO): tfrecord file to write to
        """
        for input_state, output_state in zip(self.cur_input, self.cur_output):
            data = {
                "input_state": tf.train.Feature(
                    float_list=tf.train.FloatList(value=input_state)
                ),
                "output_state": tf.train.Feature(
                    float_list=tf.train.FloatList(value=output_state)
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=data))
            tfrecord.write(example.SerializeToString())

    def extract_data(self):
        """
        Extracts state vector from the simulation run and reshapes it to be used as
        input and output for the network.
        """
        # Create lists of runs to vertical stack later
        states = self.sim.state_vec
        self.cur_input = self.concatInputs(states[0]).squeeze()
        self.cur_output = states[1:]
        if self.shuffle:
            self.shuffle_data()

    def map_fn(
        self, serialized_example: tf.train.Example
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Decodes the serialized example to tensors

        Args:
            serialized_example (tf.train.Example): serialized example to be parsed

        Returns:
            tuple[tf.Tensor, tf.Tensor]: input and output tensors
        """
        feature = {
            "input_state": tf.io.FixedLenFeature([self.input_dim], tf.float32),
            "output_state": tf.io.FixedLenFeature([self.output_dim], tf.float32),
        }
        example = tf.io.parse_single_example(serialized_example, feature)
        return example["input_state"], example["output_state"]

    def normalize(
        self, example: tf.train.Example, is_input: bool, invert: bool = False
    ) -> tf.Tensor:
        """Normalizes the given example based on known max values.

        Args:
            example (tf.Tensor): tensor to be normalized
            is_input (bool): whether the tensor is an input or output
            invert (bool, optional): bool to indicate if the tensor
            should be normalized or denormalize. Defaults to False.

        Returns:
            tf.Tensor: normalized or denormalized tensor
        """
        if is_input:
            norm_factors = self.input_norm_factors
        else:
            norm_factors = self.output_norm_factors

        dim = len(norm_factors)
        max_values_tensor = tf.constant(
            norm_factors, shape=(1, 1, dim), dtype=tf.float32
        )
        if invert:
            normalized_example = tf.multiply(example, max_values_tensor)
        else:
            normalized_example = tf.divide(example, max_values_tensor)

        normalized_example = tf.squeeze(normalized_example)

        return normalized_example


class NetworkEngine(BaseNetworkEngine):
    def __init__(self, config: dict):
        super().__init__(config)

    def get_hypermodel_fn(self) -> typing.Callable:
        """Returns a function that creates a compiled model with tunable hyperparameters

        The FF hypermodel uses an input layer and dense layers all the way through

        Returns:
            typing.Callable: function that returns a compiled model
        """
        strategy = self.choose_distribution_strategy()

        def hypermodel_fn(hp):
            keras.backend.clear_session()

            with strategy():
                model = keras.Sequential()
                model.add(
                    keras.Input(shape=(self.input_dim,), name="input_layer")
                )  # add input layer
                hp_units = hp.Int(
                    "dense_width",
                    min_value=self.width_range[0],
                    max_value=self.width_range[1],
                    step=self.width_range[2],
                )
                hp_depth = hp.Int(
                    "depth",
                    min_value=self.depth_range[0],
                    max_value=self.depth_range[1],
                    step=self.depth_range[2],
                )
                hp_act = hp.Choice("dense_act_fn", values=self.activation_fns)
                hp_reg = hp.Float(
                    "reg_param",
                    min_value=self.reg_range[0],
                    max_value=self.reg_range[1],
                    sampling=self.reg_range[2],
                )
                hp_kernel = hp.Choice("dense_kernel", values=self.kernel_inits)
                hp_bias = hp.Choice("dense_bias", values=self.bias_inits)

                # add hidden layers
                for i in range(hp_depth - 1):
                    model.add(
                        keras.layers.Dense(
                            hp_units,
                            activation=hp_act,
                            name=f"hidden_Layer_{i}",
                            kernel_initializer=hp_kernel,
                            bias_initializer=hp_bias,
                            kernel_regularizer=keras.regularizers.l2(hp_reg),
                            bias_regularizer=keras.regularizers.l2(hp_reg),
                        )
                    )
                # add output layer
                model.add(
                    keras.layers.Dense(
                        self.output_dim,
                        activation=None,
                        name="output_layer",
                        kernel_initializer=hp_kernel,
                        bias_initializer=hp_bias,
                        kernel_regularizer=keras.regularizers.l2(hp_reg),
                        bias_regularizer=keras.regularizers.l2(hp_reg),
                    )
                )

                # Model Compilation
                hp_learning_rate = hp.Choice(
                    "learning_rate", values=self.learning_rates
                )
                model.compile(
                    optimizer=self.optimizer,
                    loss=self.loss_fn,
                    metrics=self.metrics,
                )
                model.optimizer.learning_rate = hp_learning_rate
                return model

        return hypermodel_fn


class TestEngine(BaseTestEngine):
    def __init__(self, config, data_engine):
        super().__init__(config, data_engine)


class Configurator(BaseConfigurator):
    def __init__(self, config_name: str, config_dir: str = "configs"):
        """Configurator for the FF network, sets up default parameter config for network
        but the user can create new ones by calling the methods

        Args:
            config_dir (str): directory to store the configuration file
            config_name (str): name of the configuration file
        """
        super().__init__(config_dir, config_name)

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

        self.set_network(network)

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

        self.set_data(data)

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

        self.set_data(data)

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

        self.set_tuner(tuner)


def generate_default_config():
    config = Configurator("ff_default")
    config.general_network()
    config.general_data()
    config.generation_data()
    config.tuning_data()
    config.write()


if not os.path.isfile("configs/ff_default.json"):
    generate_default_config()
