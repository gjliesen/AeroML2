# Library imports
from datetime import datetime
from ..aero_ml.data_engine import DataEngine
from ..aero_ml.network_engine import NetworkEngine
from ..aero_ml.test_engine import TestEngine

# Create the directories needed to store model data
tuner_dir = "/home/ubuntu/aircraft-nn-training-data/tuner"
model_dir = "/home/ubuntu/aircraft-nn-training-data/models"
checkpoint_dir = "/home/ubuntu/aircraft-nn-training-data/model_checkpoints"
training_dir = "/home/ubuntu/aircraft-nn-training-data"

NetworkEngine.create_directories(tuner_dir, model_dir, checkpoint_dir)

date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
project_name = f"{date_str}_aircraft_sim"
batch_size = 2048

# Load the training and validation datasets
train_dataset = (
    DataEngine.load_dataset("Train", data_dir=training_dir).cache().batch(batch_size)
)
val_dataset = (
    DataEngine.load_dataset("Val", data_dir=training_dir).cache().batch(batch_size)
)

# Specifying non-default hyperparameters
loss_fn = NetworkEngine.root_mean_squared_error
activation_functions = ["relu", "tanh", "sigmoid"]
# Initialize the hypermodel function
hypermodel_fn = NetworkEngine.get_DNN_hypermodel_fn(loss_fn, activation_functions)
# tune the hypermodel
tuner = NetworkEngine.build_tuner(
    hypermodel_fn, "hyperband", project_name, max_epochs=10, directory=tuner_dir
)
tuner = NetworkEngine.tune_model(tuner, train_dataset, val_dataset, es_patience=4)

NetworkEngine.eval_tuner(tuner)

hypermodel, history = NetworkEngine.train_tuned_model(
    train_dataset, val_dataset, epochs=20, tuner=tuner, model_name="aircraft_sim_0"
)
NetworkEngine.plot_network_history(history)

hypermodel, history = NetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_1",
    tuner_trial=1,
    checkpoint_dir=checkpoint_dir,
    model_dir=model_dir,
)
NetworkEngine.plot_network_history(history)

hypermodel, history = NetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_2",
    tuner_trial=2,
    checkpoint_dir=checkpoint_dir,
    model_dir=model_dir,
)
NetworkEngine.plot_network_history(history)

hypermodel, history = NetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_3",
    tuner_trial=3,
    checkpoint_dir=checkpoint_dir,
    model_dir=model_dir,
)
NetworkEngine.plot_network_history(history)

hypermodel, history = NetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_4",
    tuner_trial=4,
    checkpoint_dir=checkpoint_dir,
    model_dir=model_dir,
)
NetworkEngine.plot_network_history(history)

test_engine = TestEngine()
test_engine.test_model(hypermodel)
