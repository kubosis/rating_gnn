from loguru import logger
import time
from typing import Callable

from ._utils import process_kwargs


def time_debug(func: Callable):
    """Decorator that reports the execution time for debugging"""

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        logger.debug(f"{func.__name__} execution took {end - start}")
        return result

    return wrap

