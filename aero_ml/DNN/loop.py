import typing
import os
import tensorflow as tf
from tensorflow import keras
from aero_ml.defaults import Defaults
from aero_ml.base.data_engine import BaseDataEngine
from aero_ml.base.network_engine import BaseNetworkEngine
from aero_ml.base.test_engine import BaseTestEngine


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
        self.cur_input = states[0:-1]
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
    def __init__(self, config: dict, data_engine: DataEngine):
        super().__init__(config, data_engine)

    def _extract_data(self, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Extract the input and output data from the dataset.

        Args:
            dataset (tf.data.Dataset): dataset containing the test data

        Returns:
            tuple[np.ndarray, np.ndarray]: tuple of the normalized input and output data
            arrays
        """
        input_data_list = []
        output_data_list = []
        for inp, outp in dataset:
            input_data_list.append(inp.numpy())
            output_data_list.append(outp.numpy())
            # Combining all the records into two numpy arrays
        # Test Input
        norm_input_arr = np.vstack(tuple(input_data_list))
        # Test Output
        norm_output_arr = np.vstack(tuple(output_data_list))

        return (norm_input_arr, norm_output_arr)

    def _process_data(self, fname: str, model_to_test) -> pd.DataFrame:
        # Looping through and building dataset from tfrecords
        dataset, _ = self.data_eng.load_dataset(fname=fname)
        # Extracting the normalized input and output array from the datset
        norm_input_arr, norm_output_arr = self._extract_data(dataset)

        # Running the inputs through the model
        input_state = norm_input_arr[0]
        norm_model_output = np.zeros((norm_output_arr.shape))
        for i in range(len(norm_output_arr)):
            # Getting the next state from the model
            output_state = model_to_test(input_state)
            # adding that state to the output array
            norm_model_output[i] = output_state
            # setting the next input to the output of the model
            input_state = output_state

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
        [true_input_arr, true_output_arr, pred_output_arr] = denorm_arrays
        # Defining truth and input dataframes
        input_df = pd.DataFrame(true_input_arr, columns=self.columns_input)

        true_output_df = pd.DataFrame(true_output_arr, columns=self.columns_output)

        # Creating dataframe for output array
        pred_output_df = pd.DataFrame(pred_output_arr, columns=self.columns_output)

        # Joining the two dataframes for plotting and comparison
        comp_df = true_output_df.join(
            pred_output_df, lsuffix="_Truth", rsuffix="_Prediction"
        )
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
        comp_df = comp_df.drop(columns=["Time_Prediction"])

        # Return dataframe for plotting
        return comp_df


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


config_name = "loop_default"
dirname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
default_config_path = os.path.join(dirname, f"{config_name}.json")
if not os.path.isfile(default_config_path):
    generate_config(config_name, dirname)
