import time
import os
from datetime import datetime


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
