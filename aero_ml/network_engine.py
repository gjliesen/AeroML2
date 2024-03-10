# Library imports
import os
from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import typing
from datetime import datetime
from aero_ml.utils import get_most_recent

Tuner = typing.Union[kt.Hyperband, kt.RandomSearch, kt.BayesianOptimization, None]


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


class NetworkEngine:
    config_name: str
    input_dim: int
    output_dim: int
    loss_fn_str: str
    optimizer: str
    metrics: list
    tuner_dir: str
    width_range: list
    depth_range: list
    activation_fns: list
    kernel_inits: list
    bias_inits: list
    learning_rates: list
    reg_range: list

    def __init__(self):
        if self.loss_fn_str == "rmse":
            self.loss_fn = root_mean_squared_error
        else:
            self.loss_fn = self.loss_fn_str
        # convertings string optimizer to actual function
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")

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
        os.makedirs(self.tuner_dir, exist_ok=True)
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
            max_trials=max_epochs,
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
