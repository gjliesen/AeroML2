# Library imports
import os
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
from datetime import datetime
from sim_data_engine import SimDataEngine


class SimNetworkEngine:
    @staticmethod
    def create_directories(
        tuner_dir="tuner", model_dir="models", checkpoint_dir="model_checkpoints"
    ):
        os.makedirs(tuner_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    @staticmethod
    def choose_distribution_strategy():
        # Check for TPU availability
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            if tpu:
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                return tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            pass  # No TPU found

        # Check for GPU availability
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Use MirroredStrategy for multiple GPUs
            if len(gpus) > 1:
                return tf.distribute.MirroredStrategy()
            # Use default strategy for a single GPU
            return tf.distribute.get_strategy()

        # Default to None when only a CPU is available
        return None

    @staticmethod
    def get_hypermodel_fn(
        loss_function="mse",
        activation_functions=["tanh", "sigmoid"],
        depth_range=[6, 12, 2],
        width_range=[20, 50, 2],
        regularization_range=[1e-11, 0.001224, "log"],
        kernel_initializer_choices=[
            "glorot_normal",
            "glorot_uniform",
            "he_normal",
            "he_uniform",
            "random_normal",
            "random_uniform",
        ],
        bias_initializer_choices=["zeros", "random_uniform"],
        learning_rate_choices=[1e-2, 1e-3, 1e-4],
        optimizer=keras.optimizers.Adam,
        metrics=["mse"],
    ):

        # Dummy context manager for use when no strategy is provided
        class dummy_context_mgr:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        strategy = SimNetworkEngine.choose_distribution_strategy()

        def hypermodel_fn(hp):
            keras.backend.clear_session()

            with strategy.scope() if strategy is not None else dummy_context_mgr():
                model = keras.Sequential()
                model.add(keras.Input(shape=(14,)))  # add input layer
                hp_units = hp.Int(
                    f"dense_width",
                    min_value=width_range[0],
                    max_value=width_range[1],
                    step=width_range[2],
                )
                hp_depth = hp.Int(
                    f"depth",
                    min_value=depth_range[0],
                    max_value=depth_range[1],
                    step=depth_range[2],
                )
                hp_act = hp.Choice(f"dense_act_fn", values=activation_functions)
                hp_reg = hp.Float(
                    f"reg_param",
                    min_value=regularization_range[0],
                    max_value=regularization_range[1],
                    sampling=regularization_range[2],
                )
                hp_kernel = hp.Choice(
                    f"dense_kernel", values=kernel_initializer_choices
                )
                hp_bias = hp.Choice(f"dense_bias", values=bias_initializer_choices)

                # add hidden layers
                for i in range(hp_depth - 1):
                    model.add(
                        keras.layers.Dense(
                            hp_units,
                            activation=hp_act,
                            name=f"Layer_{i}",
                            kernel_initializer=hp_kernel,
                            bias_initializer=hp_bias,
                            kernel_regularizer=keras.regularizers.l2(hp_reg),
                            bias_regularizer=keras.regularizers.l2(hp_reg),
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
                    "learning_rate", values=learning_rate_choices
                )
                model.compile(
                    optimizer=optimizer(learning_rate=hp_learning_rate),
                    loss=loss_function,
                    metrics=metrics,
                )
                return model

        return hypermodel_fn

    @staticmethod
    def build_tuner(
        hypermodel_fn,
        tuner_type,
        project_name,
        objective_metric="val_loss",
        objective_minmax="min",
        max_epochs=8,
        factor=3,
        directory="tuner",
        overwrite=False,
    ):
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
    def tune_model(
        tuner,
        train_dataset,
        val_dataset,
        es_metric="val_loss",
        es_patience=4,
        min_delta=0.001,
    ):
        stop_early = keras.callbacks.EarlyStopping(
            monitor=es_metric, min_delta=min_delta, patience=10
        )
        tuner.search(
            train_dataset,
            epochs=8,
            validation_data=val_dataset,
            verbose=True,
            callbacks=[stop_early],
        )
        return tuner

    @staticmethod
    def get_most_recent_tuner(folder_path):
        # Ensure the folder path is a string
        if not isinstance(folder_path, str):
            raise ValueError("folder_path must be a string")

        # List all files in the directory
        files = os.listdir(folder_path)

        # Sort files by datetime
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

        hyper_params = tuner.get_best_hyperparameters()
        for hyper_param in hyper_params:
            print(hyper_param)

    @staticmethod
    def train_tuned_model(
        train_dataset,
        val_dataset,
        epochs=10,
        tuner=None,
        path=None,
        checkpoint_dir="model_checkpoints",
        model_name=None,
        tuner_trial=0,
        es_monitor="val_loss",
        es_patience=3,
        min_delta=0.001,
        checkpoint_mode="min",
        model_dir="models",
    ):
        # Check if either tuner or path is provided, but not both
        if (tuner is None and path is None) or (tuner is not None and path is not None):
            raise ValueError('Specify either "tuner" or "path", but not both.')

        # If only path is provided, retrieve the tuner using the provided path
        if tuner is None:
            tuner = SimNetworkEngine.retrieve_tuner(path)

        if model_name is None:
            date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
            model_name = f"{date_str}_unnamed"

        hyper_params = tuner.get_best_hyperparameters()[tuner_trial]

        hypermodel = tuner.hypermodel.build(hyper_params)

        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoints = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor=es_monitor,
            mode=checkpoint_mode,
            save_best_only=True,
            verbose=1,
        )

        stop_early = keras.callbacks.EarlyStopping(
            monitor=es_monitor, min_delta=min_delta, patience=es_patience
        )

        history = hypermodel.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[checkpoints, stop_early],
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

        # # Linear plot
        # plt.figure(figsize=(20, 8))
        # plt.plot(range(epochs), history['mse'], c='r', linewidth=2)
        # plt.plot(range(epochs), history['val_loss'], c ='b', linestyle='dashed')
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend(["Training Loss", "Test Loss"])
        # plt.title("Test and Training Loss")
        # plt.show()
