import time
import os
import json
from datetime import datetime
from aero_ml.defaults import Defaults


def timeit(f):
    """Decorator to time the execution of a function"""

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return timed


def parse_datetime(filename: str) -> datetime:
    """parses the datetime strings in a folder

    Args:
        filename (str):

    Returns:
        datetime: list of datetimestrings
    """
    date_time_str = filename.split("_")[:2]
    date_time_str = "_".join(date_time_str)
    return datetime.strptime(date_time_str, "%m%d%Y_%H%M%S")


def get_most_recent(folder_path: str) -> str:
    """Retrieves the most recent file from the specified directory

    Args:
        folder_path (str): path to the directory containing the file

    Raises:
        ValueError: folder_path must be a string
        FileNotFoundError: No recent file found in the specified directory

    Returns:
        str: path to the most recent file
    """
    # Ensure the folder path is a string
    if not isinstance(folder_path, str):
        raise ValueError("folder_path must be a string")

    # List all files in the directory
    files = os.listdir(folder_path)

    # Sort files by datetime
    sorted_files = sorted(files, key=parse_datetime, reverse=True)  # type: ignore

    # Return the most recent file
    if sorted_files:
        path = os.path.join(folder_path, sorted_files[0])
        return path
    else:
        raise FileNotFoundError("No recent file found in the specified directory")


class BaseConfigurator:
    def __init__(self, config_dir: str, config_name: str):
        self.config = dict(
            config_name=config_name,
            config_dir=config_dir,
        )
        self.fname = f"{config_dir}/{config_name}.json"

    def configure(self, network=None, dirs=None, tuner=None, data=None):
        if network is not None:
            self.set_network(network)
        if dirs is not None:
            self.set_dirs(dirs)
        if tuner is not None:
            self.set_tuner(tuner)
        if data is not None:
            self.set_data(data)

    def set_network(self, network):
        self.config.update(network)

    def set_dirs(self, dirs):
        self.config.update(dirs)

    def set_tuner(self, tuner):
        self.config.update(tuner)

    def set_data(self, data):
        self.config.update(data)

    def write(self, indent=1):
        with open(self.fname, "w") as outfile:
            json.dump(self.config, outfile, indent=indent)
