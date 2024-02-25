# Library imports
from datetime import datetime
from sim_data_engine import SimDataEngine
from sim_network_engine import SimNetworkEngine
from sim_test_engine import SimTestEngine

# Create the directories needed to store model data
SimNetworkEngine.create_directories()

date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
project_name = f"{date_str}_aircraft_sim"
batch_size = 2048

# Load the training and validation datasets
train_dataset = SimDataEngine.load_dataset("Train").cache().batch(batch_size)
val_dataset = SimDataEngine.load_dataset("Val").cache().batch(batch_size)

# Specifying non-default hyperparameters
loss_fn = SimNetworkEngine.root_mean_squared_error
activation_functions = ["relu", "tanh", "sigmoid"]
# Initialize the hypermodel function
hypermodel_fn = SimNetworkEngine.get_hypermodel_fn(loss_fn, activation_functions)
# tune the hypermodel
tuner = SimNetworkEngine.build_tuner(
    hypermodel_fn, "hyperband", project_name, max_epochs=10
)
tuner = SimNetworkEngine.tune_model(tuner, train_dataset, val_dataset, es_patience=4)

SimNetworkEngine.eval_tuner(tuner)

hypermodel, history = SimNetworkEngine.train_tuned_model(
    train_dataset, val_dataset, epochs=20, tuner=tuner, model_name="aircraft_sim_0"
)
SimNetworkEngine.plot_network_history(history)

hypermodel, history = SimNetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_1",
    tuner_trial=1,
)
SimNetworkEngine.plot_network_history(history)

hypermodel, history = SimNetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_2",
    tuner_trial=2,
)
SimNetworkEngine.plot_network_history(history)

hypermodel, history = SimNetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_3",
    tuner_trial=3,
)
SimNetworkEngine.plot_network_history(history)

hypermodel, history = SimNetworkEngine.train_tuned_model(
    train_dataset,
    val_dataset,
    epochs=20,
    tuner=tuner,
    model_name="aircraft_sim_4",
    tuner_trial=4,
)
SimNetworkEngine.plot_network_history(history)

test_engine = SimTestEngine()
test_engine.test_model(hypermodel)
