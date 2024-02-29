import time
import os
import tensorflow as tf
import numpy as np
from simulation.aircraft_sim_no_forces import AircraftSimNoForces
from datetime import datetime

# pylint: disable=E1101


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return timed


class SimDataEngine:
    def __init__(self, data_config: dict):
        # dynamically etting the configuration attributes
        for key, value in data_config.items():
            setattr(self, key, value)
        self.t_vec = np.arange(0, self.run_time + self.frequency, self.frequency)

        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.meta_data = f"{self.config_name}_{self.NETWORK_TYPE}_{self.INPUT_DIM}_to_{self.OUTPUT_DIM}"
        self.cur_input = None
        self.cur_output = None

    def generate_dataset(self):
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
    def write_tfrecords(self, name: str, path, iters):
        """_summary_

        Args:
            name (str): _description_
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

    def write_example(self, tfrecord: object):
        """_summary_

        Args:
            tfrecord (object): _description_

        Raises:
            ValueError: _description_
        """
        if self.NETWORK_TYPE == "DNN":
            input_states = self.cur_input
            output_states = self.cur_output
        elif self.NETWORK_TYPE == "RNN":
            input_states = [self.cur_input]
            output_states = [self.cur_output]
        else:
            raise ValueError(
                "Invalid configuration. Supported configurations are 'DNN' and 'RNN'."
            )

        for input_state, output_state in zip(input_states, output_states):
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

    def randomize(self, input_tup: tuple):
        """_summary_

        Args:
            input_tup (tuple): _description_

        Returns:
            _type_: _description_
        """
        rng = np.random.default_rng()
        if self.rnd_method == "random":
            return rng.uniform(input_tup[0], input_tup[1])
        elif self.rnd_method == "normal":
            return rng.normal(input_tup[0], input_tup[1])
        else:
            print("unsupported method")

    def run_sim(self, itr: int):
        """_summary_

        Args:
            itr (int): _description_

        Returns:
            _type_: _description_
        """
        # Initialize the initial conditons using numpy.random.uniform
        ic_lla = np.array(
            [self.randomize(self.constraints[key]) for key in self.constraints]
        )
        # Initialize the simulation
        sim = AircraftSimNoForces(ic_lla)
        # Run the simulation
        sim.run_simulation(self.run_time, 0, 0, 0, 0, np.array([0, 0, 0]))
        if (itr + 1) % 10000 == 0:
            sim.all_states_graph("verify")
        # Extract data from simulation
        self._extract_sim_data(sim)
        if np.isnan(self.cur_output).any():
            print(self.cur_input)
            return False
        else:
            return True

    def _extract_sim_data(self, sim: object):
        """_summary_

        Args:
            sim (object): _description_
        """
        # Create lists of runs to vertical stack later
        self.cur_input = self.concatInputs(sim.lla_states_init).squeeze()
        self.cur_output = sim.get_all_states()
        if self.shuffle:
            self.shuffle_data()

    def concatInputs(self, inits: np.ndarray):
        """_summary_

        Args:
            inits (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        # Add concatInputs function from old file to stack input instances
        col_concat = np.reshape(inits, (len(inits), 1))
        arr = np.array(
            [np.vstack((self.t_vec[i], col_concat)) for i in range(len(self.t_vec))]
        )
        return arr

    def shuffle_data(self):
        """_summary_"""
        # Define indices array
        indices = np.arange(self.cur_input.shape[0])
        np.random.shuffle(indices)
        self.cur_input = self.cur_input[indices]
        self.cur_output = self.cur_output[indices]

    def map_fn(self, serialized_example):
        """_summary_

        Args:
            serialized_example (_type_): _description_

        Returns:
            _type_: _description_
        """
        feature = {
            "input_state": tf.io.FixedLenFeature([self.INPUT_DIM], tf.float32),
            "output_state": tf.io.FixedLenFeature([self.OUTPUT_DIM], tf.float32),
        }
        example = tf.io.parse_single_example(serialized_example, feature)
        return example["input_state"], example["output_state"]

    def get_most_recent_dataset(self, folder_path: str = "data") -> str:
        """Searches a given folder and dataset and looks for the most recently generated one

        Args:
            folder_path (str): path to folder to search in
            dataset_type (str): the type of dataset you're loading

        Raises:
            ValueError: dataset_type must be 'Train' or 'Val'

        Returns:
            _type_: if a files are found it will return the most recent one, otherwise it wil return nothing
        """
        # List all files in the directory
        dirs = os.listdir(folder_path)

        # Filter files based on compatible data
        filtered_dirs = [d for d in dirs if self.meta_data == d[16::]]

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
    ) -> tf.data.Dataset:
        """_summary_

        Args:
            data_dir (str, optional): _description_. Defaults to "data".
            fname (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
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
        """_summary_

        Args:
            example (object): _description_
            max_values (list): _description_
            invert (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if is_input:
            norm_factors = self.INPUT_NORM_FACTORS
        else:
            norm_factors = self.OUTPUT_NORM_FACTORS

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
    def parse_datetime(filename: str):
        """parses the datetime strings in a folder

        Args:
            filename (str):

        Returns:
            _type_: list of datetimestrings
        """
        date_time_str = filename.split("_")[:2]
        date_time_str = "_".join(date_time_str)
        return datetime.strptime(date_time_str, "%m%d%Y_%H%M%S")
