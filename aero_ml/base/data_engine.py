import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime
from aero_ml.utils import timeit, get_most_recent
from aero_ml.simulation.aircraft_sim import AircraftSim


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

        self.meta_data = f"{self.config_name}-{self.input_dim}-{self.output_dim}"
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
        print("Data Engine Initialized")

    def generate_dataset(self, shuffle=None):
        """Function to generate the training and test datasets for the network."""
        if shuffle is not None:
            self.shuffle = shuffle
        self.length = 0

        # Initialize the simulation object
        self.sim = AircraftSim()

        # Creating training dataset path
        dir_name = f"{self.date_str}-{self.meta_data}"
        train_path = Path.cwd().joinpath("data", dir_name, "train.tfrecord")
        train_path.parent.mkdir(parents=True, exist_ok=True)

        # Write training datasets to folder
        self.write_tfrecords(train_path, self.iterations)
        print(self.length)

        # Create folder for test data
        test_path = train_path.parent.joinpath("test")
        test_path.mkdir(parents=True, exist_ok=True)

        # setting shuffle back to false for test data
        self.shuffle = False
        # Write test datasets to folder
        for case in range(self.test_cases):
            tmp_path = test_path.joinpath(f"test_{case}.tfrecord")
            self.write_tfrecords(tmp_path, 1)

    @timeit
    def write_tfrecords(self, path: Path, iters: int):
        """creates a directory for  the given path, runs the simulation for iters number
        of times and writes the output to a tfrecord file

        currently writing after each simulation to prevent issues with memory

        Args:
            name (str): file name
            path (os.PathLike): path to write the file
            iters (int): number of times to run the simulation
        """
        print(f"starting write to {path.name}...")
        with tf.io.TFRecordWriter(str(path)) as tfrecord:
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
        assert self.sim.t_vec is not None
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

    def load_dataset(
        self, input_path: os.PathLike
    ) -> tuple[tf.data.Dataset, os.PathLike]:
        """Takes a directory or a filename, then loads and preprocesses a tfrecords
        dataset

        Args:
            data_path (str, optional): Can be a filename or a directory, if it's a
            filename it will be loaded otherwise the directory will be searched for a
            compatible file. Defaults to Path("data").

        Returns:
            tuple[tf.data.Dataset, str]: _description_
        """
        # convert input_path to pathlib object
        input_path = Path(input_path)
        # loading data if path to file else find the most recent file in the directory
        if input_path.is_dir():
            file_path = get_most_recent(input_path, f"*{self.meta_data}/train.tfrecord")
        else:
            file_path = input_path

        data_path = file_path.parent
        # load the tf record dataset, parse the examples and normalize the data
        dataset = tf.data.TFRecordDataset(str(file_path))
        dataset = dataset.map(self.map_fn)
        dataset = dataset.map(
            lambda input_state, output_state: (
                self.normalize(input_state, True),
                self.normalize(output_state, False),
            )
        )

        return (dataset, data_path)

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
