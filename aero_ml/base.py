import os
import typing
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from datetime import datetime
from aero_ml.defaults import Defaults
from aero_ml.utils import timeit, parse_datetime, get_most_recent
from aero_ml.simulation.aircraft_sim import AircraftSim, quat_to_euler


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


class BaseDataEngine:
    config_name: str  # name of the configuration
    config_dir: str  # directory of the configuration
    input_dim: int  # input dimension of the network
    output_dim: int  # output dimension of the network
    run_time: float  # how long a simulation runs in seconds
    frequency: float  # simulation time step in hz
    iterations: int  # number of iterations to run the simulation
    test_cases: int  # number of test cases to run
    rnd_method: str  # method for randomizing initial conditions
    input_norm_factors: list  # list of max values for input normalization
    output_norm_factors: list  # list of max values for output normalization
    constraints: dict  # dictionary of constraints for randomization
    shuffle: bool = False  # whether to shuffle the data

    def __init__(self, config: dict):
        # Dynamically unpack the configuration dictionary into class attributes
        valid_keys = [
            "config_name",
            "config_dir",
            "input_dim",
            "output_dim",
            "run_time",
            "frequency",
            "iterations",
            "test_cases",
            "rnd_method",
            "input_norm_factors",
            "output_norm_factors",
            "constraints",
            "shuffle",
        ]
        for key in valid_keys:
            setattr(self, key, config[key])

        self.meta_data = f"{self.config_name}_{self.input_dim}_to_{self.output_dim}"
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
        print("Data Engine Initialized")

    def generate_dataset(self):
        """Function to generate the training and test datasets for the network."""

        # Initialize the simulation object
        self.sim = AircraftSim()
        # Training dataset
        dir_name = f"{self.date_str}_{self.meta_data}"

        train_path = os.path.join(os.getcwd(), f"data/{dir_name}")
        os.makedirs(train_path, exist_ok=True)
        self.write_tfrecords("train", train_path, self.iterations)

        # Create folder for test data
        test_path = os.path.join(train_path, "test")
        os.makedirs(test_path, exist_ok=True)

        # Write test datasets to folder
        for case in range(self.test_cases):
            self.write_tfrecords(f"test_{case}", test_path, 1)

    @timeit
    def write_tfrecords(self, name: str, path: str, iters: int):
        """creates a directory for  the given path, runs the simulation for iters number
        of times and writes the output to a tfrecord file

        currently writing after each simulation to prevent issues with memory

        Args:
            name (str): file name
            path (os.PathLike): path to write the file
            iters (int): number of times to run the simulation
        """
        print(f"starting write {name} records...")
        record_path = os.path.join(path, f"{name}.tfrecord")
        with tf.io.TFRecordWriter(record_path) as tfrecord:
            for itr in range(iters):
                valid = self.run_sim(itr)
                if valid:
                    self.write_example(tfrecord)  # type: ignore
                if (itr + 1) % 500 == 0:
                    print(f"iteration: {itr+1} of {iters}")

    def write_example(self, tfrecord: tf.io.TFRecordWriter):
        """Write example to tfrecord file

        Args:
            tfrecord (tf.io.TFRecordWriter): tfrecord file to write to
        """
        raise NotImplementedError("write_example must be implemented in subclass")

    def randomize(self, input_tup: tuple[float, float]) -> float:
        """randomizes the initial conditions based on the given constraints

        Args:
            input_tup (tuple[float, float]): tuple of min and max values for the
            randomization if the method is "random" or the mean and standard deviation
            if the method is "normal"

        Returns:
            float: the randomized value
        """
        rng = np.random.default_rng()

        if self.rnd_method == "random":
            rnd_fn = rng.uniform
        elif self.rnd_method == "normal":
            rnd_fn = rng.normal
        else:
            raise ValueError("rnd_method must be 'random' or 'normal'")

        return rnd_fn(input_tup[0], input_tup[1])

    def run_sim(self, itr: int) -> bool:
        """Generate the initial conditions, run the simulation, and extract the data

        Args:
            itr (int): the current iteration of the simulation

        Returns:
            bool: Boolean indicating if the simulation output is valid
        """
        # Initialize the initial conditons using numpy.random.uniform
        ic_arr = np.array(
            [self.randomize(self.constraints[key]) for key in self.constraints]
        )

        # Decompose the initial condition array
        p_E_E_BE_0 = ic_arr[0:3]
        att_B_B_BN_0 = ic_arr[3:6]
        v_E_B_BE_0 = ic_arr[6:9]
        w_B_B_BE_0 = ic_arr[9:]

        # Initialize the simulation
        self.sim.set_initial_conditions(
            p_E_E_BE_0, att_B_B_BN_0, v_E_B_BE_0, w_B_B_BE_0
        )
        # Run the simulation
        self.sim.run_simulation(self.run_time, frequency=self.frequency)

        # # Defining an update frequency for verifying sim output
        # if (itr + 1) % 1000 == 0:
        #     self.sim.all_states_graph("verify")

        # Extract data from simulation
        # these methods need to return arrays rather than alter member variables
        self.extract_data()

        # Skipping simulation output if it's empty
        if np.isnan(self.cur_output).any():
            print(self.cur_input)
            return False
        else:
            return True

    def extract_data(self):
        """
        Extracts state vector from the simulation run and reshapes it to be used as
        input and output for the network.
        """
        raise NotImplementedError("extract_sim_data must be implemented in subclass")

    def concatInputs(self, inits: np.ndarray) -> np.ndarray:
        """Concatenates the input data with the time vector

        Args:
            inits (np.ndarray): numpy array of initial conditions

        Returns:
            np.ndarray: numpy array of initial conditions with time vector
        """
        # Add concatInputs function from old file to stack input instances
        col_concat = np.reshape(inits, (len(inits), 1))
        arr = np.array(
            [
                np.vstack((self.sim.t_vec[i], col_concat))
                for i in range(len(self.sim.t_vec))
            ]
        )
        return arr

    def shuffle_data(self):
        """Shuffles the input and output data."""
        # Define indices array
        indices = np.arange(self.cur_input.shape[0])
        np.random.shuffle(indices)
        self.cur_input = self.cur_input[indices]
        self.cur_output = self.cur_output[indices]

    def map_fn(
        self, serialized_example: tf.train.Example
    ) -> tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("map_fn must be implemented in subclass")

    def get_most_recent_dataset(self, folder_path: str = "data") -> str:
        """Searches a given folder and dataset and looks for the most recently
        generated one

        Args:
            folder_path (str): path to folder to search in
            dataset_type (str): the type of dataset you're loading

        Raises:
            ValueError: dataset_type must be 'Train' or 'Val'

        Returns:
            str: if a files are found it will return the most recent one, otherwise
            it wil return nothing
        """
        # List all files in the directory
        dirs = os.listdir(folder_path)

        # Filter files based on compatible data
        filtered_dirs = [d for d in dirs if self.meta_data == d[16:]]

        # Sort files by datetime
        sorted_dirs = sorted(filtered_dirs, key=parse_datetime, reverse=True)

        # check if the sorted files list is empty, raising error if true
        if not sorted_dirs:
            raise FileNotFoundError("No compatible files were found")

        # Return the most recent file
        return f"{folder_path}/{sorted_dirs[0]}"

    def load_dataset(
        self,
        fname: str = "",
        search_dir: str = "data",
    ) -> tuple[tf.data.Dataset, str]:
        """_summary_

        Args:
            fname (str, optional): _description_. Defaults to "".
            search_dir (str, optional): _description_. Defaults to "data".

        Returns:
            tuple[tf.data.Dataset, str]: _description_
        """
        # loading data if fname is provided otherwise searching for the most recent
        if fname == "":
            data_dir = self.get_most_recent_dataset(search_dir)
            fname = os.path.join(data_dir, "train.tfrecord")
        else:
            data_dir = ""
        # load the tf record dataset, parse the examples and normalize the data
        dataset = tf.data.TFRecordDataset(fname)
        dataset = dataset.map(self.map_fn)
        dataset = dataset.map(
            lambda input_state, output_state: (
                self.normalize(input_state, True),
                self.normalize(output_state, False),
            )
        )

        return (dataset, data_dir)

    def normalize(self, example: object, is_input: bool, invert=False):
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


# Define a tuner type
Tuner = typing.Union[kt.Hyperband, kt.RandomSearch, kt.BayesianOptimization, None]


# saving RMSE as a custom loss function
@keras.saving.register_keras_serializable(name="rmse")
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(
        tf.reduce_mean(tf.square(y_pred - y_true))
    )  # noinspection PyArgumentList


# Dummy context manager for use when no strategy is provided
class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class BaseNetworkEngine:
    """
    Base class for creating and tuning neural networks, contains methods for
    hyperparameter tuning and model training. The class is designed to be inherited
    into one subclass for eacch specific network architecture, and the member
    variables are to be set in the subclass. The subclass overrides several methods
    which raise value errors if they are not overwritten
    """

    config_name: str  # name of the configuration
    input_dim: int  # number of input features
    output_dim: int  # number of output features
    loss_fn_str: str  # string representation of the loss function
    optimizer: str  # string representation of the optimizer
    metrics: list  # list of metrics to use
    tuner_dir: str  # directory to save the tuner
    width_range: list  # list of possible widths for the network
    depth_range: list  # list of possible depths for the network
    activation_fns: list  # list of possible activation functions
    kernel_inits: list  # list of possible kernel initializers
    bias_inits: list  # list of possible bias initializers
    learning_rates: list  # list of possible learning rates
    reg_range: list  # list of possible regularization values

    def __init__(self, config: dict):
        """Initializes the NetworkEngine class, sets the loss function to a callable if
        it is a custom loss function, otherwise the string is kept as is. Also sets the
        date string to the current date and time"""
        # Dynamically unpack the configuration dictionary into class attributes
        valid_keys = [
            "config_name",
            "input_dim",
            "output_dim",
            "loss_fn_str",
            "optimizer",
            "metrics",
            "width_range",
            "depth_range",
            "activation_fns",
            "kernel_inits",
            "bias_inits",
            "learning_rates",
            "reg_range",
        ]
        for key in valid_keys:
            setattr(self, key, config[key])

        if self.loss_fn_str == "rmse":
            self.loss_fn = root_mean_squared_error
        else:
            self.loss_fn = self.loss_fn_str
        # convertings string optimizer to actual function
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
        print("Network Engine initialized")

    def choose_distribution_strategy(self) -> typing.Callable:
        """Detects and returns the appropriate distribution strategy for the current
        environment.

        Returns:
            typing.Callable: strategy context manager
        """
        # Check for TPU availability
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            if tpu:
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                return tf.distribute.experimental.TPUStrategy(tpu).scope
        except ValueError:
            pass  # No TPU found

        # Check for GPU availability
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Use MirroredStrategy for multiple GPUs
            if len(gpus) > 1:
                return tf.distribute.MirroredStrategy().scope
            # Use default strategy for a single GPU
            return tf.distribute.get_strategy().scope
        return dummy_context_mgr

    def get_hypermodel_fn(self) -> typing.Callable:
        """Returns a function that creates a compiled model with tunable hyperparameters

        Returns:
            typing.Callable: function that returns a compiled model
        """
        raise NotImplementedError("get_hypermodel_fn must be implemented in subclass")

    def build_tuner(
        self,
        hypermodel_fn: typing.Callable,
        tuner_type: str,
        objective_metric: str = "val_loss",
        objective_minmax: str = "min",
        max_epochs: int = 8,
        directory: str = "tuner",
        overwrite: bool = False,
        project_name: str = "",
    ) -> Tuner:
        """Builds a keras tuner object

        Args:
            hypermodel_fn (typing.Callable): function that returns a compiled model
            tuner_type (str): type of tuner to use
            objective_metric (str, optional): objective metric to optimize. Defaults to
            "val_loss".
            objective_minmax (str, optional): sets whether tuner objective is to
            minimize or maximize the metric. Defaults to "min".
            max_epochs (int, optional): maximum number of epochs to try. Defaults to 8.
            factor (int, optional): tuning factor to use. Defaults to 3.
            directory (str, optional): directory to save the tuner. Defaults to "tuner".
            overwrite (bool, optional): whether to overwrite the tuner if it exists.
            project_name (str, optional): name of the project. Defaults to "".

        Raises:
            ValueError: If an unsupported tuner type is provided by the user

        Returns:
            Tuner: Keras tuner object to be used for hyperparameter tuning
        """
        os.makedirs(directory, exist_ok=True)
        if project_name == "":
            project_name = f"{self.date_str}_{self.config_name}"
        # Define the objective
        objective = kt.Objective(objective_metric, objective_minmax)

        if tuner_type == "hyperband":
            tuner_cls = kt.Hyperband
        elif tuner_type == "randomsearch":
            tuner_cls = kt.RandomSearch
        elif tuner_type == "bayesian":
            tuner_cls = kt.BayesianOptimization
        else:
            raise ValueError(f"Unsupported tuner type: {tuner_type}")

        tuner = tuner_cls(
            hypermodel_fn,
            objective=objective,
            max_epochs=max_epochs,
            directory=directory,
            project_name=project_name,
            overwrite=overwrite,
        )
        return tuner

    def get_most_recent_tuner(self, folder_path: str) -> Tuner:
        """Retrieves the most recent tuner from the specified directory

        Args:
            folder_path (str): path to the directory containing the tuner

        Raises:
            ValueError: folder_path must be a string
            FileNotFoundError: No recent tuner found in the specified directory

        Returns:
            Tuner: Keras tuner object
        """
        # Ensure the folder path is a string
        if not isinstance(folder_path, str):
            raise ValueError("folder_path must be a string")

        path = get_most_recent(folder_path)

        # Return the most recent file
        return kt.load_tuner(path)

    def retrieve_tuner(self, path: str = "") -> Tuner:
        """Retrieves a tuner from the specified path or the most recent tuner from the
        tuner directory

        Args:
            path (str, optional): path to the tuner. Defaults to "".

        Returns:
            Tuner: Keras tuner object
        """
        if path == "":
            tuner = self.get_most_recent_tuner("tuner")
        else:
            tuner = kt.load_tuner(path)
        return tuner

    def eval_tuner(self, tuner: Tuner = None, path: str = ""):
        """prints the top ten hyperparmeter combinations from the tuner

        Args:
            tuner (Tuner, optional): Keras tuner object. Defaults to None.
            path (str, optional): path to the tuner. Defaults to "".

        Raises:
            ValueError: Specify either "tuner" or "path", but not both.
        """
        # Check if either tuner or path is provided, but not both
        if (tuner is None and path == "") or (tuner is not None and path != ""):
            raise ValueError('Specify either "tuner" or "path", but not both.')

        # If only path is provided, retrieve the tuner using the provided path
        if tuner is None:
            tuner = self.retrieve_tuner(path)

        hyper_params = tuner.get_best_hyperparameters(num_trials=10)  # type: ignore

        for hyper_param in hyper_params:
            print(hyper_param.values)

    def train_tuned_model(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        callbacks: list,
        epochs: int = 10,
        tuner: Tuner = None,
        path: str = "",
        tuner_trial: int = 0,
        model_dir: str = "models",
    ) -> tuple[keras.Model, dict]:
        """Trains a model using the best hyperparameters found by the tuner

        Args:
            train_dataset (tf.data.Dataset): training dataset
            val_dataset (tf.data.Dataset): validation dataset
            callbacks (list): list of callbacks to use during training
            epochs (int, optional): number of epochs to train. Defaults to 10.
            tuner (Tuner, optional): Keras tuner object. Defaults to None.
            path (str, optional): path to the tuner. Defaults to "".
            tuner_trial (int, optional): trial number to use. Defaults to 0.
            model_dir (str, optional): directory to save the model. Defaults to "models".

        Raises:
            ValueError: Specify either "tuner" or "path", but not both.

        Returns:
            tuple[keras.Model, dict]: trained model and history
        """
        # Check if either tuner or path is provided, but not both
        if (tuner is None and path == "") or (tuner is not None and path != ""):
            raise ValueError('Specify either "tuner" or "path", but not both.')

        # If only path is provided, retrieve the tuner using the provided path
        if tuner is None:
            tuner = self.retrieve_tuner(path)

        model_name = f"{self.date_str}_{self.config_name}"

        hyper_params = tuner.get_best_hyperparameters(num_trials=10)[tuner_trial]

        hypermodel = tuner.hypermodel.build(hyper_params)

        history = hypermodel.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=True,
        )

        hypermodel.save(f"{model_dir}/{model_name}.keras")

        return (hypermodel, history)

    def plot_network_history(self, history):
        """Plots the training and validation loss of the network in a log scale"""
        epochs = history["epoch"]
        # Log plot
        plt.figure(figsize=(20, 8))
        plt.semilogy(range(epochs), history["mse"], c="r", linewidth=2)
        plt.semilogy(range(epochs), history["val_loss"], c="b", linestyle="dashed")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Training Loss", "Test Loss"])
        plt.title("Test and Training Loss")
        plt.show()


class BaseTestEngine:
    """
    class for testing the model on the test data and plotting the results.
    """

    date_str: str
    input_features_euler: list
    output_features_euler: list
    input_features: list
    output_features: list

    def __init__(
        self,
        config,
        data_eng,
        att_mode: str = "Euler",
    ):
        valid_keys = [
            "input_features_euler",
            "output_features_euler",
            "input_features",
            "output_features",
        ]
        for key in valid_keys:
            setattr(self, key, config[key])
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.data_eng = data_eng
        # dynamically etting the configuration attributes
        self.attitude_mode = att_mode
        self.columns_input = []
        self.columns_output = []
        self._set_columns()

        self.plot_columns = []
        self.line_dicts = {}
        self._set_figure_params()

        print("Test Engine Initialized")

    def _set_columns(self):
        """Set the input and output columns based on the attitude mode."""
        if self.attitude_mode == "Euler":
            self.columns_input = self.input_features_euler
            self.columns_output = self.output_features_euler
        else:
            self.columns_input = self.input_features
            self.columns_output = self.output_features

    def _set_figure_params(self):
        col_types = ["_Truth", "_Prediction"]
        colors = px.colors.qualitative.Dark24
        for i, col in enumerate(self.columns_output):
            subplot = []
            for col_type in col_types:
                full_name = f"{col}{col_type}"
                subplot.append(full_name)
                if col_type == "_Truth":
                    self.line_dicts[full_name] = dict(color=colors[i])
                else:
                    self.line_dicts[full_name] = dict(color=colors[i], dash="dash")
            self.plot_columns.append(subplot)

    def test_model(self, model_to_test, data_dir: str):
        """Test the model on the test data and plot the results.

        Args:
            model_to_test (_type_): tensorflow model to be tested
            data_dir (str): directory containing the test data
        """
        # Defining the directories
        test_dir = f"{data_dir}/test"
        plot_dir = f"test_data_plots/{data_dir}"
        os.makedirs(plot_dir, exist_ok=True)

        # Placing all test data records into a list
        files = os.listdir(test_dir)

        # Looping through test_data and plotting
        for fname in files:
            # Define test record file name, joining the path
            tfrecord_path = f"{test_dir}/{fname}"
            # Creating a comparison dataframe to plot
            comp_df = self._process_data(tfrecord_path, model_to_test)
            # Plotting comparison dataframe
            self.test_plots(comp_df, fname, plot_dir)

    def _process_data(self, fname: str, model_to_test) -> pd.DataFrame:
        """Process the test data and return a dataframe for plotting.

        Args:
            fname (str): file name of the test data
            model_to_test (_type_): tensorflow model to be tested

        Returns:
            pd.DataFrame: dataframe of the test and predicted data used for plotting
        """
        # Looping through and building dataset from tfrecords
        dataset, _ = self.data_eng.load_dataset(fname=fname)
        # Extracting the normalized input and output array from the datset
        norm_input_arr, norm_output_arr = self._extract_data(dataset)

        # Running the inputs through the model
        norm_model_output = model_to_test(norm_input_arr)

        norm_arrays = [norm_input_arr, norm_output_arr, norm_model_output]
        denorm_arrays = self._denormalize_data(norm_arrays)
        if self.attitude_mode == "Euler":
            denorm_arrays = self._convert_attitude(denorm_arrays)

        return self._create_dataframe(denorm_arrays)

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

    def _denormalize_data(self, norm_arrays: list[np.ndarray]) -> list[np.ndarray]:
        """Denormalize the input and output arrays.

        Args:
            norm_arrays (list[np.ndarray]): list of the normalized input and output
            arrays

        Returns:
            list[np.ndarray]: list of the denormalized input and output arrays
        """
        is_inputs = [True, False, False]
        denorm_arrays = []
        for arr, is_input in zip(norm_arrays, is_inputs):
            denorm_arrays.append(
                (
                    self.data_eng.normalize(arr, is_input=is_input, invert=True)
                    .numpy()
                    .squeeze()
                )
            )
        return denorm_arrays

    def _convert_attitude(self, denorm_arrays: list[np.ndarray]) -> list[np.ndarray]:
        """wrapper function for _convert_quaternions that loops through the arrays that
        need to be converted.

        Args:
            denorm_arrays (list[np.ndarray]): list of the denormalized input and output
            arrays

        Returns:
            list[np.ndarray]: list of the denormalized input and output arrays with
            euler angles rather than quaternions
        """
        # Convert attitude to euler angles if selected
        quat_indices = [4, 3, 3]
        euler_arrays = []
        for arr, quat_idx in zip(denorm_arrays, quat_indices):
            euler_arrays.append(self._convert_quaternions(arr, quat_idx))
        return euler_arrays

    def _convert_quaternions(self, arr: np.ndarray, q_start: int) -> np.ndarray:
        """Applies the quaternion to euler conversion to the given array and returns an
        array of euler angles.

        Args:
            arr (np.ndarray): array containing the quaternions
            q_start (int): starting index of the quaternions

        Returns:
            np.ndarray: array containing the euler angles
        """
        # Ending index of quaternions
        q_end = q_start + 4
        # Euler angle conversion
        euler_angles = np.apply_along_axis(quat_to_euler, 1, arr[0:, q_start:q_end])
        # It is only position if an output array is passed
        time_and_pos = arr[0:, 0:q_start]
        vel_and_body_rates = arr[0:, q_end:]
        # Return new array
        return np.hstack((time_and_pos, euler_angles, vel_and_body_rates))

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
        # Adding time to output array for simplicity
        true_output_df["Time"] = input_df["Time"]

        # Creating dataframe for output array
        pred_output_df = pd.DataFrame(pred_output_arr, columns=self.columns_output)

        # Adding time for simplicty
        pred_output_df["Time"] = input_df["Time"]

        # Joining the two dataframes for plotting and comparison
        comp_df = true_output_df.join(
            pred_output_df, lsuffix="_Truth", rsuffix="_Prediction"
        )
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
        comp_df = comp_df.drop(columns=["Time_Prediction"])

        # Return dataframe for plotting
        return comp_df

    def test_plots(
        self, comp_df: pd.DataFrame, title: str, plot_dir: str, height: int = 6000
    ):
        """Plot the test data and the model predictions.

        Args:
            comp_df (pd.DataFrame): dataframe of the test and predicted data
            title (str): title of the plot
            plot_dir (str): directory to save the plot
            height (int, optional): height of the plot in pixels. Defaults to 6000.
        """
        # Initializing subplot
        fig = make_subplots(
            rows=len(self.plot_columns),
            cols=1,
            shared_xaxes=False,
            subplot_titles=tuple(self.columns_output),
        )

        # Looping through the column sets to plot
        for row, col_list in enumerate(self.plot_columns):
            for col in col_list:
                fig.add_trace(
                    go.Scatter(
                        x=comp_df["Time_Truth"],
                        y=comp_df[col],
                        name=col,
                        line=self.line_dicts[col],
                        legendgroup=str(row + 1),
                    ),
                    row=row + 1,
                    col=1,
                )
        # Displaying the figure
        fig.update_layout(hovermode="x unified", title=title, height=height)
        fig.write_html(f"{plot_dir}/{title}.html")
