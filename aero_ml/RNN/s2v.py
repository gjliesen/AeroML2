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

        S2V RNN uses tensors in the input and output so the numpy arrays are
        serialized to bytes list

        Args:
            tfrecord (typing.IO): tfrecord file to write to
        """
        for input_state, output_state in zip(self.cur_input, self.cur_output):
            states_input = tf.convert_to_tensor(input_state, dtype=tf.float32)
            states_output = tf.convert_to_tensor(output_state, dtype=tf.float32)

            serialized_input = tf.io.serialize_tensor(states_input)
            serialized_output = tf.io.serialize_tensor(states_output)

            data = {
                "input_state": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[serialized_input.numpy()])
                ),
                "output_state": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[serialized_output.numpy()])
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

        # cur_input and cur_output are lists of numpy arrays
        self.cur_input = []
        self.cur_output = []
        for i in range(len(states) - 1):
            # the input state is a sequence from the initial condition to i
            self.cur_input.append(states[:i])
            # the output state is the next state after the input state
            self.cur_output.append(states[i + 1])

    def map_fn(
        self, serialized_example: tf.train.Example
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Decodes the serialized example to tensors

        The RNN uses tensors as input and output so the serialized tensors are parsed
        from a bytes string to a tensors

        Args:
            serialized_example (tf.train.Example): serialized example to be parsed

        Returns:
            tuple[tf.Tensor, tf.Tensor]: input and output tensors
        """
        feature = {
            "input_state": tf.io.FixedLenFeature([], tf.string),
            "output_state": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature)

        input_state = tf.io.parse_tensor(example["input_state"], out_type=tf.float32)
        output_state = tf.io.parse_tensor(example["output_state"], out_type=tf.float32)

        return (input_state, output_state)

    def normalize(
        self, example: tf.Tensor, is_input: bool, invert: bool = False
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

        normalized_example = tf.squeeze(normalized_example, axis=0)

        return normalized_example


class NetworkEngine(BaseNetworkEngine):
    def __init__(self, config: dict):
        super().__init__(config)

    def build_RNN(self, width):
        keras.backend.clear_session()
        # Define the distribution strategy for training
        scope = self.choose_distribution_strategy()

        input_shape = [None, 1, self.input_dim]
        with scope():
            model = keras.models.Sequential()
            model.add(
                keras.layers.LSTM(
                    width,
                    return_sequences=True,
                    input_shape=input_shape,
                )
            )
            model.add(keras.layers.LSTM(width))
            model.add(keras.layers.Dense(13))
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                metrics=self.metrics,
            )
            return model

    def get_hypermodel_fn(self) -> typing.Callable:
        strategy = self.choose_distribution_strategy()

        def hypermodel_fn(hp):
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
            # hp_reg = hp.Float(
            #     f"reg_param",
            #     min_value=self.reg_range[0],
            #     max_value=self.reg_range[1],
            #     sampling=self.reg_range[2],
            # )
            hp_kernel = hp.Choice("dense_kernel", values=self.kernel_inits)
            hp_bias = hp.Choice("dense_bias", values=self.bias_inits)
            with strategy():
                keras.backend.clear_session()
                # Intialize model
                model = keras.models.Sequential()

                # add input layer
                input_shape = [None, self.input_dim]
                model.add(
                    keras.layers.LSTM(
                        units=hp_units,
                        return_sequences=True,
                        input_shape=input_shape,
                        activation=hp_act,
                        kernel_initializer=hp_kernel,
                        bias_initializer=hp_bias,
                    )
                )
                # add hidden layers
                for i in range(hp_depth - 2):
                    model.add(
                        keras.layers.LSTM(
                            units=hp_units,
                            return_sequences=True,
                            activation=hp_act,
                            kernel_initializer=hp_kernel,
                            bias_initializer=hp_bias,
                        )
                    )

                model.add(
                    keras.layers.LSTM(
                        units=hp_units,
                        activation=hp_act,
                        kernel_initializer=hp_kernel,
                        bias_initializer=hp_bias,
                    )
                )
                # add output layer
                model.add(keras.layers.Dense(13))

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
    def __init__(self, config: dict, data_engine: DataEngine):
        super().__init__(config, data_engine)


class Configurator(BaseConfigurator):
    config: dict

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
        input_dim: int = 13,
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
        input_dict = maximums_quat

        [*input_features], [*input_norm_factors] = zip(*input_dict.items())

        # Configuring DNN outputs
        output_dict = maximums_quat
        [*output_features], [*output_norm_factors] = zip(*output_dict.items())

        data = dict(
            input_features=input_features,
            input_features_euler=features_euler,
            input_norm_factors=input_norm_factors,
            output_features=output_features,
            output_features_euler=features_euler,
            output_norm_factors=output_norm_factors,
        )

        self.config.update(data)


def generate_config(config_name: str, config_dir: str):
    config = Configurator(config_name, config_dir)
    config.general_network()
    config.general_data()
    config.generation_data()
    config.tuning_data()
    config.write()


config_name = "s2v_default"
dirname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
default_config_path = os.path.join(dirname, f"{config_name}.json")
if not os.path.isfile(default_config_path):
    generate_config(config_name, dirname)
