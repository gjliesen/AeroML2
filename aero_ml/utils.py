import time
import os
from typing import Union
from pathlib import Path


def timeit(f):
    """Decorator to time the execution of a function"""

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return timed


def get_most_recent(folder_path: Union[os.PathLike, str], filter: str = "*") -> Path:
    """Retrieves the most recent file from the specified directory

    Args:
        folder_path (Path): path to the directory containing the file
        filter (str, optional): filter for the file. Defaults to "*".
    Raises:
        ValueError: folder_path must be a string
        FileNotFoundError: No recent file found in the specified directory

    Returns:
        Path: path to the most recent file
    """
    # Conversion incase folder_path is a string
    folder_path = Path(folder_path)
    # Sort files by datetime
    sorted_paths = sorted(folder_path.glob(filter), reverse=True)

    # Return the most recent file
    if sorted_paths:
        return sorted_paths[0]
    else:
        raise FileNotFoundError("No recent file found in the specified directory")
