import os
import json
import typing
import tensorflow as tf


class Manager:
    config_path: str  # The path to the configuration file
    data_engine: typing.Callable  # the data engine for the network, will be a subclass
    network_engine: typing.Callable  # the network engine, will be a subclass
    test_engine: typing.Callable  # The test engine, will be a subclass
    model_dir: str  # The directory to save the model
    checkpoint_dir: str  # The directory to save the model checkpoints
    tuner_dir: str  # The directory to save the tuner
    config: dict  # Configuration for the overrall network
    data_dir: str  # Directory where the loaded data is stored
    dataset: tf.data.Dataset  # The full dataset
    train_dataset: tf.data.Dataset  # The training dataset
    val_dataset: tf.data.Dataset  # The validation dataset

    def __init__(
        self,
        config_path: str,
        data_engine: typing.Callable,
        network_engine: typing.Callable,
        test_engine: typing.Callable,
        model_dir: str = "models",
        checkpoint_dir: str = "checkpoints",
        tuner_dir: str = "tuners",
    ):
        self.config = self.load_config(config_path)
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.tuner_dir = tuner_dir
        # Initialize the engines
        self.data_engine = data_engine(self.config)
        self.network_engine = network_engine(self.config)
        self.test_engine = test_engine(self.config, self.data_engine)

    def create_dirs(self):
        """Create the directories for the model, checkpoints, and tuner"""
        for dir in [self.model_dir, self.checkpoint_dir, self.tuner_dir]:
            os.makedirs(dir, exist_ok=True)

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as infile:
            return json.load(infile)

    def generate_dataset(self):
        """wrapper for the data engine generate_dataset method"""
        self.data_engine.generate_dataset()

    def load_dataset(self, search_dir: str = "data"):
        """wrapper for the data engine load_dataset method"""
        dataset, data_dir = self.data_engine.load_dataset(search_dir=search_dir)
        self.dataset = dataset
        self.data_dir = data_dir

    def split_dataset(self, val_pct: float = 0.2):
        """
        Split the dataset into training and validation sets.

        Args:
            val_pct (float): The percentage of the dataset to use for validation
        """
        val_length = int(val_pct * self.config["length"])
        self.val_dataset = self.dataset.take(val_length)
        self.train_dataset = self.dataset.skip(val_length)

    def set_batch_size(self, batch_size: int = 10000):
        """
        Set the batch size for the training and validation sets.

        Args:
            batch_size (int): The batch size to use
        """
        self.val_dataset = self.val_dataset.batch(batch_size)
        self.train_dataset = self.train_dataset.batch(batch_size)

    def tune_network(self, epochs: int = 4, tuner_type: str = "hyperband"):
        """
        Tune the network hyperparameters.

        Args:
            epochs (int): The number of epochs to use for tuning
        """
        hypermodel_fn = self.network_engine.get_hypermodel_fn()
        self.tuner = self.network_engine.build_tuner(hypermodel_fn, tuner_type)
        self.tuner.search(
            self.train_dataset, epochs=epochs, validation_data=self.val_dataset
        )

    def train_tuned_network(self, epochs: int = 10, callbacks: list = []):
        """
        Train the network using the tuned hyper parameters

        Args:
            epochs (int): The number of epochs to train for
        """
        self.hypermodel, self.history = self.network_engine.train_tuned_model(
            self.train_dataset, self.val_dataset, callbacks, epochs, self.tuner
        )

    def test_model(self):
        self.test_engine.test_model(self.hypermodel, self.data_dir)
