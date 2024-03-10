import time
from datetime import datetime


def timeit(f):
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
