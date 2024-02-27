import time
import os
import tensorflow as tf
import numpy as np
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
        constraints,
        rate,
        time,
        iterations,
        rnd_method="random",
        title="sim_output",
        shuffle=False,
    ):
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.constraints = constraints
        self.title = title
        self.rate = rate
        self.time = time
        self.iterations = iterations
        self.rnd_method = rnd_method
        self.t_vec = np.arange(0, self.time + self.rate, self.rate)[5:-10]
        self.shuffle = shuffle

    @timeit
    def create_dataset(self, purpose="train"):
        """Primary function, used to call the aircraft simulation and write the output to a tfrecords file, it will either be a test or training dataset, this determines the folder to write the data to

        Args:
            purpose (str, optional): _description_. Defaults to "train".
        """
        print("starting to write records...")
        if purpose == "Train":
            dir = "training_data"
        else:
            dir = "test_data"

        with tf.io.TFRecordWriter(
            f"{dir}/{self.date_str}_{self.title}_{purpose}.tfrecord"
        ) as tfrecord:
            for iter in self.iterations:
                self.run_sim(iter)
                if np.isnan(self.tmp_output).any():
                    print(self.tmp_input)
                    continue
                else:
                    pass
                if (iter + 1) % 500 == 0:
                    print(f"iteration: {iter+1} of {self.iterations}")

    def write_example(tfrecord, input_state, output_state):
        for i in range(len(input_state)):
            data = {
                "input_state": tf.train.Feature(
                    float_list=tf.train.FloatList(value=input_state[i])
                ),
                "output_state": tf.train.Feature(
                    float_list=tf.train.FloatList(value=output_state[i])
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=data))
        tfrecord.write(example.SerializeToString())

    def create_training_set(
        self,
        constraints,
        rate,
        time,
        iters,
        rnd_method="random",
        title="sim_output",
        shuffle=True,
    ):

        inputs = []
        outputs = []
        for iter in range(self.iterations):
            self.run_sim(iter)
            if np.isnan(self.tmp_output).any():
                print(self.tmp_input)
                continue
            else:
                inputs.append(self.tmp_input)
                outputs.append(self.tmp_output)
            if (iter + 1) % 500 == 0:
                print(f"iteration: {iter+1} of {self.iterations}")
        # Store the numpy versions of the training data in a list
        input_states = np.vstack(tuple(inputs))
        output_states = np.vstack(tuple(outputs))

        if self.shuffle:
            training_split = 0.8
            split_idx = int(np.floor(len(input_states) * training_split))

            # Write train data
            self._write_records(
                input_states[0:split_idx], output_states[0:split_idx], "Train"
            )

            # Write validation data
            self._write_records(
                input_states[split_idx::], output_states[split_idx::], "Val"
            )
        else:
            # Write train data
            self._write_records(input_states, output_states, "Test")

    def randomize(self, input_tup):
        rng = np.random.default_rng()
        if self.rnd_method == "random":
            return rng.uniform(input_tup[0], input_tup[1])
        elif self.rnd_method == "normal":
            return rng.normal(input_tup[0], input_tup[1])
        else:
            print("unsupported method")

    def run_sim(self, iter):
        # Initialize the initial conditons using numpy.random.uniform
        ic_lla = np.array(
            [self.randomize(self.constraints[key]) for key in self.constraints]
        )
        # Initialize the simulation
        sim = AircraftSimNoForces(ic_lla)
        # Run the simulation
        sim.run_simulation(self.time, 0, 0, 0, 0, np.array([0, 0, 0]))
        if (iter + 1) % 10000 == 0:
            sim.all_states_graph("verify")
        # Extract data from simulation
        self._extract_sim_data(sim)

    def _extract_sim_data(self, sim):
        # Create lists of runs to vertical stack later
        self.tmp_input = self.concatInputs(sim.lla_states_init).squeeze()[5:-10]
        self.tmp_output = sim.get_all_states()[5:-10]
        if self.shuffle:
            self.shuffle_data()

    # @timeit
    def concatInputs(self, inits):
        # Add concatInputs function from old file to stack input instances
        col_concat = np.reshape(inits, (len(inits), 1))
        arr = np.array(
            [np.vstack((self.t_vec[i], col_concat)) for i in range(len(self.t_vec))]
        )
        return arr

    def shuffle_data(self):
        # Define indices array
        indices = np.arange(self.tmp_input.shape[0])
        np.random.shuffle(indices)
        self.tmp_input = self.tmp_input[indices]
        self.tmp_output = self.tmp_output[indices]

    @staticmethod
    def map_fn(serialized_example, input_len=14, output_len=13):
        feature = {
            "input_state": tf.io.FixedLenFeature([input_len], tf.float32),
            "output_state": tf.io.FixedLenFeature([output_len], tf.float32),
        }
        example = tf.io.parse_single_example(serialized_example, feature)
        return example["input_state"], example["output_state"]

    @staticmethod
    def normalize_data(example, max_values, invert=False):
        dim = len(max_values)
        max_values_tensor = tf.constant(max_values, shape=(1, 1, dim), dtype=tf.float32)
        if invert:
            normalized_example = tf.multiply(example, max_values_tensor)
        else:
            normalized_example = tf.divide(example, max_values_tensor)
        normalized_example = tf.squeeze(normalized_example)
        return normalized_example

    @staticmethod
    def parse_datetime(filename):
        date_time_str = filename.split("_")[:2]
        date_time_str = "_".join(date_time_str)
        return datetime.strptime(date_time_str, "%m%d%Y_%H%M%S")

    @staticmethod
    def get_most_recent_dataset(folder_path, dataset_type):
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
        purpose,
        dir="training_data",
        fname=None,
        input_norm_factors=[
            10,
            90,
            90,
            12000,
            1,
            1,
            1,
            1,
            60,
            10,
            20,
            12,
            12,
            12,
        ],
        output_norm_factors=[
            90,
            90,
            12000,
            1,
            1,
            1,
            1,
            60,
            20,
            20,
            12,
            12,
            12,
        ],
    ):
        datagen = SimDataEngine()
        if fname is None:
            fname = SimDataEngine.get_most_recent_dataset(dir, purpose)

        dataset = tf.data.TFRecordDataset(fname)
        dataset = dataset.map(datagen.map_fn)
        dataset = dataset.map(
            lambda input_state, output_state: (
                datagen.normalize_data(input_state, input_norm_factors),
                datagen.normalize_data(output_state, output_norm_factors),
            )
        )
        return dataset
