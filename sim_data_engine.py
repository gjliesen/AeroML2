import pstats
import time
import json
import os
import tensorflow as tf
import numpy as np
from numpy.typing import ArrayLike
from simulation.aircraft_sim_no_forces import AircraftSimNoForces
from datetime import datetime


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return timed


class SimDataEngine:
    def __init__(
        self,
        config_name: str,
        constraints: dict,
        frequency: float,
        run_time: int,
        iterations: int,
        test_cases: int,
        length: int,
        rnd_method: str,
        shuffle: bool,
        INPUT_DIM: int,
        OUTPUT_DIM: int,
        NETWORK_TYPE: str,
    ):
        self.config_name = config_name
        self.constraints = constraints
        self.frequency = frequency
        self.time = run_time
        self.iterations = iterations
        self.test_cases = test_cases
        self.length = length
        self.rnd_method = rnd_method
        self.shuffle = shuffle
        self.t_vec = np.arange(0, run_time + frequency, frequency)

        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM
        self.NETWORK_TYPE = NETWORK_TYPE

        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")

        self.data_path = os.path.join(os.getcwd(), "data")
        self.initialize_directories()

        self.cur_input = None
        self.cur_output = None

    def initialize_directories(self):
        os.makedirs(self.data_path, exist_ok=True)
        folder_name = f"{self.date_str}_{self.config_name}_{self.NETWORK_TYPE}_{self.INPUT_DIM}_to_{self.OUTPUT_DIM}"
        self.data_path = os.path.join(self.data_path, folder_name)
        os.makedirs(self.data_path, exist_ok=True)

    def generate_dataset(self):
        self.write_tfrecords("train")
        for case in range(self.test_cases):
            self.write_tfrecords(f"test_{case}")

    @timeit
    def write_tfrecords(self, name: str):
        """Primary function, used to call the aircraft simulation and write the output to a tfrecords file, it will either be a test or training dataset, this determines the folder to write the data to

        Args:
            purpose (str, optional): _description_. NETWORK to "train".
        """
        print(f"starting write {name} records...")
        record_path = os.path.join(self.data_path, f"{name}.tfrecords")
        with tf.io.TFRecordWriter(record_path) as tfrecord:
            for itr in self.iterations:
                valid = self.run_sim(itr)
                if valid:
                    self.write_example(tfrecord)
                if (itr + 1) % 500 == 0:
                    print(f"iteration: {itr+1} of {self.iterations}")

    def write_example(self, tfrecord: object):
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
        rng = np.random.default_rng()
        if self.rnd_method == "random":
            return rng.uniform(input_tup[0], input_tup[1])
        elif self.rnd_method == "normal":
            return rng.normal(input_tup[0], input_tup[1])
        else:
            print("unsupported method")

    def run_sim(self, itr: int):
        # Initialize the initial conditons using numpy.random.uniform
        ic_lla = np.array(
            [self.randomize(self.constraints[key]) for key in self.constraints]
        )
        # Initialize the simulation
        sim = AircraftSimNoForces(ic_lla)
        # Run the simulation
        sim.run_simulation(self.time, 0, 0, 0, 0, np.array([0, 0, 0]))
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
        # Create lists of runs to vertical stack later
        self.cur_input = self.concatInputs(sim.lla_states_init).squeeze()
        self.cur_output = sim.get_all_states()
        if self.shuffle:
            self.shuffle_data()

    # @timeit
    def concatInputs(self, inits: ArrayLike):
        # Add concatInputs function from old file to stack input instances
        col_concat = np.reshape(inits, (len(inits), 1))
        arr = np.array(
            [np.vstack((self.t_vec[i], col_concat)) for i in range(len(self.t_vec))]
        )
        return arr

    def shuffle_data(self):
        # Define indices array
        indices = np.arange(self.cur_input.shape[0])
        np.random.shuffle(indices)
        self.cur_input = self.cur_input[indices]
        self.cur_output = self.cur_output[indices]

    @staticmethod
    def map_fn(serialized_example: object, input_len: int = 14, output_len: int = 13):
        feature = {
            "input_state": tf.io.FixedLenFeature([input_len], tf.float32),
            "output_state": tf.io.FixedLenFeature([output_len], tf.float32),
        }
        example = tf.io.parse_single_example(serialized_example, feature)
        return example["input_state"], example["output_state"]

    @staticmethod
    def normalize_data(example: object, max_values: list, invert=False):
        dim = len(max_values)
        max_values_tensor = tf.constant(max_values, shape=(1, 1, dim), dtype=tf.float32)
        if invert:
            normalized_example = tf.multiply(example, max_values_tensor)
        else:
            normalized_example = tf.divide(example, max_values_tensor)
        normalized_example = tf.squeeze(normalized_example)
        return normalized_example

    @staticmethod
    def parse_datetime(filename: str):
        date_time_str = filename.split("_")[:2]
        date_time_str = "_".join(date_time_str)
        return datetime.strptime(date_time_str, "%m%d%Y_%H%M%S")

    @staticmethod
    def get_most_recent_dataset(folder_path: str, dataset_type: str):
        if dataset_type not in ["Train", "Val"]:
            raise ValueError("dataset_type must be 'Train' or 'Val'")

        # Ensure the folder path is a string
        if not isinstance(folder_path, str):
            raise ValueError("folder_path must be a string")

        # List all files in the directory
        files = os.listdir(folder_path)

        # Filter files based on the dataset type
        filtered_files = [f for f in files if dataset_type in f]

        # Sort files by datetime
        sorted_files = sorted(
            filtered_files, key=SimDataEngine.parse_datetime, reverse=True
        )

        # Return the most recent file
        return f"{folder_path}/{sorted_files[0]}" if sorted_files else None

    @staticmethod
    def load_dataset(
        input_norm_factors: list,
        output_norm_factors: list,
        dataset_type: str,
        data_dir: str = "training_data",
        fname: str = "",
    ):
        if fname == "":
            fname = SimDataEngine.get_most_recent_dataset(data_dir, dataset_type)

        dataset = tf.data.TFRecordDataset(fname)
        dataset = dataset.map(SimDataEngine.map_fn)
        dataset = dataset.map(
            lambda input_state, output_state: (
                SimDataEngine.normalize_data(input_state, input_norm_factors),
                SimDataEngine.normalize_data(output_state, output_norm_factors),
            )
        )
        return dataset
