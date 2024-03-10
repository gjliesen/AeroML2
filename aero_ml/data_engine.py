import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from aero_ml.utils import timeit
from simulation.aircraft_sim import AircraftSim


class DataEngine:
    run_time: float
    frequency: float
    iterations: int
    test_cases: int
    meta_data: str
    rnd_method: str
    input_norm_factors: list
    output_norm_factors: list
    constraints: dict
    shuffle: bool = False

    def __init__(
        self,
    ):
        self.t_vec = np.arange(0, self.run_time + self.frequency, self.frequency)

        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")

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
                    self.write_example(tfrecord)
                if (itr + 1) % 500 == 0:
                    print(f"iteration: {itr+1} of {iters}")

    def write_example(self, tfrecord):
        """Write example to tfrecord file

        FF uses lists as inputs and outputs so the inputs and outputs are just
        Args:
            tfrecord (typing.IO): tfrecord file to write to
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
        self.sim.run_simulation(self.run_time, 0, 0, 0, 0, np.array([0, 0, 0]))

        # # Defining an update frequency for verifying sim output
        # if (itr + 1) % 1000 == 0:
        #     self.sim.all_states_graph("verify")

        # Extract data from simulation
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
            [np.vstack((self.t_vec[i], col_concat)) for i in range(len(self.t_vec))]
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
            _type_: if a files are found it will return the most recent one, otherwise
            it wil return nothing
        """
        # List all files in the directory
        dirs = os.listdir(folder_path)

        # Filter files based on compatible data
        filtered_dirs = [d for d in dirs if self.meta_data == d[16:]]

        # Sort files by datetime
        sorted_dirs = sorted(filtered_dirs, key=self.parse_datetime, reverse=True)

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
        if fname == "":
            data_dir = self.get_most_recent_dataset(search_dir)
            fname = f"{data_dir}/train.tfrecord"
        else:
            data_dir = ""
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

    @staticmethod
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
