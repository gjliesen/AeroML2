# pyright: reportAttributeAccessIssue=false
import os
import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from typing import Union
from aero_ml.defaults import Defaults
from aero_ml.base.manager import BaseManager
from aero_ml.base.data_engine import BaseDataEngine
from aero_ml.base.network_engine import BaseNetworkEngine
from aero_ml.base.test_engine import BaseTestEngine
from aero_ml.base.configurator import BaseConfigurator


class S2VConfigurator(BaseConfigurator):
    config: dict

    def __init__(self, config_path: os.PathLike, config_name: str = ""):
        """Configurator for the FF network, sets up default parameter config for network
        but the user can create new ones by calling the methods

        Args:
            config_dir (str): directory to store the configuration file
            config_name (str): name of the configuration file
        """
        super().__init__(config_path, config_name)

    def general_network(
        self,
        input_dim: int = 13,
        output_dim: int = 13,
        optimizer: str = "adam",
        metrics: list[str] = ["mse"],
        loss_fn_str: str = "mse",
    ):
        super().general_network()
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
        super().general_data()
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

    def generation_data(
        self,
        run_time: int = 1,
        frequency: float = 0.01,
        iterations: int = 200000,
        test_cases: int = 30,
        constraints: dict = Defaults.constraints,
        rnd_method: str = "random",
        shuffle: bool = False,
    ):
        super().generation_data()
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


config_name = "s2v_default"
s2v_default_config_path = Path(__file__).parent / "configs" / f"{config_name}.json"
if not s2v_default_config_path.exists():
    cfg = S2VConfigurator(s2v_default_config_path)
    cfg.generate()


class S2VDataEngine(BaseDataEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.seq_len = int(config["run_time"] / config["frequency"])

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
            self.length += 1

    def extract_data(self):
        """
        Extracts state vector from the simulation run and reshapes it to be used as
        input and output for the network.
        """
        # Create lists of runs to vertical stack later
        states = self.sim.state_vec
        assert states is not None
        # cur_input and cur_output are lists of numpy arrays
        self.cur_input = []
        self.cur_output = []
        max_length = len(states)
        for i in range(1, max_length):
            # the input state is a sequence from the initial condition to i
            padded_states = np.zeros((max_length - 1, states.shape[1]))
            padded_states[-i:] = states[:i]

            self.cur_input.append(padded_states)
            # the output state is the next state after the input state
            self.cur_output.append(states[i])  # noqa: E203

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

        input_state = tf.reshape(input_state, (self.seq_len, 13))
        output_state = tf.reshape(output_state, (1, 13))
        return input_state, output_state

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
            dim = len(norm_factors)
            max_values_tensor = tf.constant(
                norm_factors, shape=(1, 1, dim), dtype=tf.float32
            )

        else:
            norm_factors = self.output_norm_factors
            dim = len(norm_factors)
            max_values_tensor = tf.constant(
                norm_factors, shape=(1, dim), dtype=tf.float32
            )

        if invert:
            normalized_example = tf.multiply(example, max_values_tensor)
        else:
            normalized_example = tf.divide(example, max_values_tensor)

        if len(normalized_example.shape) == 3 and normalized_example.shape[0] == 1:
            normalized_example = tf.squeeze(normalized_example, axis=0)

        return normalized_example


class S2VNetworkEngine(BaseNetworkEngine):
    def __init__(self, config: dict):
        super().__init__(config)

    def build_model(self, width):
        keras.backend.clear_session()
        # Define the distribution strategy for training
        scope = self.choose_distribution_strategy()

        with scope():
            model = keras.models.Sequential()
            model.add(
                keras.layers.LSTM(
                    width,
                    return_sequences=True,
                    input_shape=(None, self.input_dim),
                )
            )
            model.add(
                keras.layers.LSTM(
                    width,
                )
            )
            model.add(keras.layers.Dense(13))
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                metrics=self.metrics,
            )
            return model

    def get_hypermodel_fn(self) -> typing.Callable:
        scope = self.choose_distribution_strategy()

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
            with scope():
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
                        return_sequences=False,
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


class S2VTestEngine(BaseTestEngine):
    def __init__(
        self, config: dict, data_engine: S2VDataEngine, att_mode: str = "Euler"
    ):
        super().__init__(config, data_engine, att_mode)

    def _extract_data(self, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Extract the input and output data from the dataset.

        Args:
            dataset (tf.data.Dataset): dataset containing the test data

        Returns:
            tuple[np.ndarray, np.ndarray]: tuple of the normalized input and output data
            arrays
        """
        output_data_list = []
        for inp, outp in dataset:  # type: ignore
            output_data_list.append(outp.numpy())
            # Combining all the records into two numpy arrays
        # Test Input
        norm_input_arr = inp.numpy()
        # Test Output
        norm_output_arr = np.vstack(tuple(output_data_list))

        if norm_input_arr.shape != norm_output_arr.shape:
            raise ValueError(
                "The input and output data shapes do not match. "
                f"Input shape: {norm_input_arr.shape}, "
                f"Output shape: {norm_output_arr.shape}"
            )
        return (norm_input_arr, norm_output_arr)

    def _process_data(self, path: Path, model_to_test) -> pd.DataFrame:
        # Looping through and building dataset from tfrecords
        dataset, _ = self.data_eng.load_dataset(path)
        # Extracting the normalized input and output array from the datset
        norm_input_arr, norm_output_arr = self._extract_data(dataset)

        # Running the inputs through the model
        # initializing the model output array
        norm_model_output = np.zeros((norm_output_arr.shape))
        norm_model_input = np.zeros((1, *norm_output_arr.shape))
        # setting the first input to the first input of the test data
        norm_model_output[0] = norm_input_arr[0]
        for i in range(1, len(norm_output_arr)):
            # growing the sequence of inputs at each time step
            norm_model_input[0, -i:] = norm_model_output[:i]
            # Getting the next state from the model
            print(norm_model_input)
            output_state = model_to_test(norm_model_input)
            # adding that state to the output array
            norm_model_output[i] = output_state

        norm_arrays = [norm_input_arr, norm_output_arr, norm_model_output]
        denorm_arrays = self._denormalize_data(norm_arrays)
        if self.attitude_mode == "Euler":
            denorm_arrays = self._convert_attitude(denorm_arrays)

        return self._create_dataframe(denorm_arrays)

    def _create_dataframe(self, denorm_arrays: list[np.ndarray]) -> pd.DataFrame:
        """Create a dataframe for plotting from the input, output, and predicted arrays.

        Args:
            denorm_arrays (list[np.ndarray]): list of the denormalized input and output

        Returns:
            pd.DataFrame: dataframe of the input, output, and predicted arrays
        """
        _, true_output_arr, pred_output_arr = denorm_arrays
        # Defining truth and input dataframes
        # input_df = pd.DataFrame(true_input_arr, columns=self.columns_input)

        true_output_df = pd.DataFrame(true_output_arr, columns=self.columns_output)

        # Creating dataframe for output array
        pred_output_df = pd.DataFrame(pred_output_arr, columns=self.columns_output)

        t0 = 0.0
        dt = 0.01
        t_vec = np.arange(t0, t0 + dt * len(true_output_arr), dt)
        # Joining the two dataframes for plotting and comparison
        comp_df = true_output_df.join(
            pred_output_df, lsuffix="_Truth", rsuffix="_Prediction"
        )
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
        comp_df["Time_Truth"] = t_vec

        # Return dataframe for plotting
        return comp_df


class S2VManager(BaseManager):
    def __init__(
        self,
        config_path: Union[os.PathLike, str],
        model_dir: Union[os.PathLike, str] = "models",
        checkpoint_dir: Union[os.PathLike, str] = "checkpoints",
        tuner_dir: Union[os.PathLike, str] = "tuners",
        att_mode: str = "euler",
    ):
        super().__init__(config_path, model_dir, checkpoint_dir, tuner_dir, att_mode)
        self.data_engine = S2VDataEngine(self.config)
        self.network_engine = S2VNetworkEngine(self.config)
        self.test_engine = S2VTestEngine(self.config, self.data_engine, att_mode)
