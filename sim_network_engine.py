# Library imports
import os
from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from datetime import datetime
from sim_data_engine import SimDataEngine

# pylint: disable=E1101


@keras.saving.register_keras_serializable(name="rmse")
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# Dummy context manager for use when no strategy is provided
class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class SimNetworkEngine:
    def __init__(self, network_config):
        for key, value in network_config.items():
            setattr(self, key, value)
        # convertings string optimizer to actual function
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")

    def choose_distribution_strategy(self):
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

    def build_RNN(self, width):
        keras.backend.clear_session()
        # Define the distribution strategy for training
        scope = self.choose_distribution_strategy()

        input_shape = [None, self.INPUT_DIM]
        with scope():
            model = keras.models.Sequential()
            model.add(
                keras.layers.LSTM(
                    width,
                    return_sequences=True,
                    input_shape=input_shape,
                )
            )
            model.add(keras.layer.RepeatVector((self.run_time / self.frequency)))
            model.add(keras.layers.LSTM(width, return_sequences=True))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(width)))
            model.compile(
                optimizer=self.optimizer,
                loss="rmse",
                metrics=self.metrics,
            )
            return model

    def get_hypermodel_fn(self):
        strategy = self.choose_distribution_strategy()
        if self.NETWORK_TYPE == "DNN":

            def hypermodel_fn(hp):
                keras.backend.clear_session()

                with strategy():
                    model = keras.Sequential()
                    model.add(
                        keras.Input(shape=(self.INPUT_DIM,), name="input_layer")
                    )  # add input layer
                    hp_units = hp.Int(
                        f"dense_width",
                        min_value=self.width_range[0],
                        max_value=self.width_range[1],
                        step=self.width_range[2],
                    )
                    hp_depth = hp.Int(
                        f"depth",
                        min_value=self.depth_range[0],
                        max_value=self.depth_range[1],
                        step=self.depth_range[2],
                    )
                    hp_act = hp.Choice(f"dense_act_fn", values=self.activation_fns)
                    hp_reg = hp.Float(
                        f"reg_param",
                        min_value=self.reg_range[0],
                        max_value=self.reg_range[1],
                        sampling=self.reg_range[2],
                    )
                    hp_kernel = hp.Choice(f"dense_kernel", values=self.kernel_inits)
                    hp_bias = hp.Choice(f"dense_bias", values=self.bias_inits)

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
                            self.OUTPUT_DIM,
                            activation=None,
                            name=f"output_layer",
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

        elif self.NETWORK_TYPE == "RNN":

            def hypermodel_fn(hp):
                keras.backend.clear_session()

                with strategy():
                    model = keras.Sequential()

                    model.add(
                        keras.layers.SimpleRNN(
                            self.INPUT_DIM,
                            input_shape=(None, self.INPUT_DIM),
                            name="input_layer",
                        )
                    )  # add input layer

                    hp_units = hp.Int(
                        f"dense_width",
                        min_value=self.width_range[0],
                        max_value=self.width_range[1],
                        step=self.width_range[2],
                    )
                    hp_depth = hp.Int(
                        f"depth",
                        min_value=self.depth_range[0],
                        max_value=self.depth_range[1],
                        step=self.depth_range[2],
                    )
                    hp_act = hp.Choice(f"dense_act_fn", values=self.activation_fns)
                    hp_reg = hp.Float(
                        f"reg_param",
                        min_value=self.reg_range[0],
                        max_value=self.reg_range[1],
                        sampling=self.reg_range[2],
                    )
                    hp_kernel = hp.Choice(f"dense_kernel", values=self.kernel_inits)
                    hp_bias = hp.Choice(f"dense_bias", values=self.bias_inits)

                    # add hidden layers
                    for i in range(hp_depth - 1):
                        model.add(
                            keras.layers.SimpleRNN(
                                hp_units,
                                return_sequences=True,
                                activation=hp_act,
                                name=f"recurrent_Layer_{i}",
                                kernel_initializer=hp_kernel,
                                bias_initializer=hp_bias,
                            )
                        )
                    # add output layer
                    model.add(
                        keras.layers.Dense(
                            13,
                            activation=None,
                            name=f"output_layer",
                            kernel_initializer=hp_kernel,
                            bias_initializer=hp_bias,
                            kernel_regularizer=keras.regularizers.l2(hp_reg),
                            bias_regularizer=keras.regularizers.l2(hp_reg),
                        )
                    )

                    # Model Compilation
                    hp_learning_rate = hp.Choice(
                        "learning_rate", values=self.learning_rate_choices
                    )
                    model.compile(
                        optimizer=self.optimizer,
                        loss=self.loss_fn,
                        metrics=self.metrics,
                    )
                    model.optimizer.learning_rate = hp_learning_rate
                    return model

        return hypermodel_fn

    def build_tuner(
        self,
        hypermodel_fn,
        tuner_type,
        objective_metric="val_loss",
        objective_minmax="min",
        max_epochs=8,
        factor=3,
        directory="tuner",
        overwrite=False,
    ):
        os.makedirs(self.tuner_dir, exist_ok=True)
        project_name = f"{self.date_str}_{self.config_name}"
        # Define the objective
        objective = kt.Objective(objective_metric, objective_minmax)

        if tuner_type == "hyperband":
            tuner = kt.Hyperband(
                hypermodel_fn,
                objective=objective,
                max_epochs=max_epochs,
                factor=factor,
                directory=directory,
                project_name=project_name,
                overwrite=overwrite,
            )
        elif tuner_type == "randomsearch":
            tuner = kt.RandomSearch(
                hypermodel_fn,
                objective=objective,
                max_trials=max_epochs,
                directory=directory,
                project_name=project_name,
                overwrite=overwrite,
            )
        elif tuner_type == "bayesian":
            tuner = kt.BayesianOptimization(
                hypermodel_fn,
                objective=objective,
                max_trials=max_epochs,
                directory=directory,
                project_name=project_name,
                overwrite=overwrite,
            )
        else:
            raise ValueError("Unsupported tuner type: " + tuner_type)

        return tuner

    @staticmethod
    def get_most_recent_tuner(folder_path: str):
        # Ensure the folder path is a string
        if not isinstance(folder_path, str):
            raise ValueError("folder_path must be a string")

        # List all files in the directory
        files = os.listdir(folder_path)

        # Sort files by datetime
        # pylint: disable=E0602
        # Pylance: disable=reportUndefinedVariable
        sorted_files = sorted(files, key=SimDataEngine.parse_datetime, reverse=True)

        # Return the most recent file
        if sorted_files:
            path = os.path.join(folder_path, sorted_files[0])
            return kt.load_tuner(path)
        else:
            raise FileNotFoundError("No recent tuner found in the specified directory")

    @staticmethod
    def retrieve_tuner(path=None):
        if path is None:
            tuner = SimNetworkEngine.get_most_recent_tuner("tuner")
        else:
            tuner = kt.load_tuner(path)
        return tuner

    @staticmethod
    def eval_tuner(tuner=None, path=None):
        # Check if either tuner or path is provided, but not both
        if (tuner is None and path is None) or (tuner is not None and path is not None):
            raise ValueError('Specify either "tuner" or "path", but not both.')

        # If only path is provided, retrieve the tuner using the provided path
        if tuner is None:
            tuner = SimNetworkEngine.retrieve_tuner(path)

        hyper_params = tuner.get_best_hyperparameters(num_trials=10)
        for hyper_param in hyper_params:
            print(hyper_param.values)

    def train_tuned_model(
        self,
        train_dataset,
        val_dataset,
        callbacks,
        epochs=10,
        tuner=None,
        path=None,
        tuner_trial=0,
        model_dir="models",
    ):
        # Check if either tuner or path is provided, but not both
        if (tuner is None and path is None) or (tuner is not None and path is not None):
            raise ValueError('Specify either "tuner" or "path", but not both.')

        # If only path is provided, retrieve the tuner using the provided path
        if tuner is None:
            tuner = SimNetworkEngine.retrieve_tuner(path)

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

    @staticmethod
    def plot_network_history(history):
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
