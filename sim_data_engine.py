import time
import os
import tensorflow as tf
import numpy as np
from simulation.aircraft_sim import Aircraft_Sim
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
        self.sim = Aircraft_Sim()

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
                    if self.NETWORK_TYPE == "DNN":
                        self.write_DNN_example(tfrecord)
                    elif self.NETWORK_TYPE == "RNN":
                        self.write_RNN_example(tfrecord)
                if (itr + 1) % 500 == 0:
                    print(f"iteration: {itr+1} of {iters}")

    def write_DNN_example(self, tfrecord: object):
        """_summary_

        Args:
            tfrecord (object): _description_

        Raises:
            ValueError: _description_
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

    def write_RNN_example(self, tfrecord: object):
        """_summary_

        Args:
            tfrecord (object): _description_

        Raises:
            ValueError: _description_
        """
        data = {
            "input_state": tf.train.Feature(
                float_list=tf.train.FloatList(value=self.cur_input.flatten())
            ),
            "output_state": tf.train.Feature(
                float_list=tf.train.FloatList(value=self.cur_output.flatten())
            ),
            "input_shape": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.cur_input.shape))
            ),
            "output_shape": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.cur_output.shape))
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
        self._extract_sim_data()

        # Skipping simulation output if it's empty
        if np.isnan(self.cur_output).any():
            print(self.cur_input)
            return False
        else:
            return True

    def _extract_sim_data(self):
        """_summary_

        Args:
            sim (object): _description_
        """
        # Create lists of runs to vertical stack later
        states = self.sim.state_vec
        if self.NETWORK_TYPE == "DNN":
            self.cur_input = self.concatInputs(states[0]).squeeze()
            self.cur_output = states
        else:
            self.cur_input = states[0].squeeze()
            self.cur_output = states[1:]
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

    def DNN_map_fn(self, serialized_example):
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

    def RNN_map_fn(self, serialized_example):
        """_summary_

        Args:
            serialized_example (_type_): _description_

        Returns:
            _type_: _description_
        """
        feature = {
            "input_state": tf.io.FixedLenFeature([self.INPUT_DIM], tf.float32),
            "output_state": tf.io.FixedLenFeature([100, self.OUTPUT_DIM], tf.float32),
            "input_shape": tf.io.FixedLenFeature([1], tf.int64),
            "output_shape": tf.io.FixedLenFeature([2], tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, feature)

        input_state = tf.reshape(example["input_state"], example["input_shape"])
        output_state = tf.reshape(example["output_state"], example["output_shape"])

        return input_state, output_state

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
        if self.NETWORK_TYPE == "DNN":
            map_fn = self.DNN_map_fn
        elif self.NETWORK_TYPE == "RNN":
            map_fn = self.RNN_map_fn
        dataset = dataset.map(map_fn)
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
