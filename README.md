# Aero ML: Integrating ML Architectures with Nonlinear 6DOF Simulation
Overview

Aero ML is a Python package designed to test various machine learning architectures as surrogate models for a nonlinear 6DOF (six degrees of freedom) simulation of an ASW-28 glider. It provides essential tools and modules for implementing Deep Neural Networks (DNN) and Recurrent Neural Networks (RNN) in the context of 6DOF aerospace simulations.


## Usage

```python
from aero_ml.manager import Manager
from aero_ml.RNN.s2v import DataEngine, NetworkEngine, TestEngine, default_config_path

"""
default_config_path is the default configuration for the imported network, custom configurations can be created with the network specific
configuration model
"""

# Create a Manager instance
manager = Manager('path_to_config.json', DataEngine, NetworkEngine, TestEngine)

# Load configuration
config = manager.load_config('path_to_config.json')

# Generate and load dataset
manager.generate_dataset()
manager.load_dataset('data_directory')

# Split and cache dataset
manager.split_dataset(val_pct=0.2)
manager.cache_dataset()

# Set batch size
manager.set_batch_size(batch_size=10000)

# Retrieve tuner
manager.retrieve_tuner('path_to_tuner')
```