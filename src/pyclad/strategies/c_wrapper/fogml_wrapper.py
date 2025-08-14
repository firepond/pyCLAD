import ctypes
import logging
import os
from typing import Dict

import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptAwareStrategy,
    ConceptIncrementalStrategy,
)

logger = logging.getLogger(__name__)


# c wrappers, using ctypes
script_dir = os.path.dirname(os.path.abspath(__file__))
so_file = os.path.join(script_dir, './obj/test.so')
# Load the shared library (.so for linux/mac, .dll for windows)
_sum = ctypes.CDLL(so_file)
# Set argument types for the function in the shared library '_sum'
# ctypes.POINTER(ctypes.c_int) is the python ctypes equivalent of an int * in C
# ctypes will check if the arguments we pass the the function fit these types
_sum.simple_threesum.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
_sum.sum_function.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int))

# Define the python function wrapper for sum_function()
def sum_function(numbers):
    #State that the library is global to make it easier to find
    global _sum
    # Query the input array for its length to pass to sum_function
    num_numbers = len(numbers)
    # Define a ctypes array type compatible with the library
    # achieved by multipying ctypes.c_int with an integer
    array_type = ctypes.c_int * num_numbers
    # This type wants every element of the array as an input (so that it can be summed) 
    # and so we pass '*numbers'
    result = _sum.sum_function(ctypes.c_int(num_numbers), array_type(*numbers))
    # The C function returns an integer, and we require a python integer as we leave the function
    # and so we return an int() of the output
    return int(result)

def simple_threesum(a, b, c):
    result = _sum.simple_threesum(ctypes.c_int(a), ctypes.c_int(b), ctypes.c_int(c))
    return int(result)

class FogmlStrategyWrapper(
    ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy
):
    def __init__(self, model: Model, max_buffer_size: int = 1000):
        self._replay = []
        self._model = model
        self.max_buffer_size = max_buffer_size
        self.buffer_size = 0

    def update(self, data: np.ndarray):
        """Update the replay buffer with new data."""
        for sample in data:
            if self.buffer_size < self.max_buffer_size:
                self._replay.append(sample)
                self.buffer_size += 1
            else:
                index = int(np.random.randint(0, self.max_buffer_size))
                self._replay[index] = sample

    def learn(self, data: np.ndarray, *_args, **_kwargs) -> None:
        """Learn from the data and store it in the replay buffer."""
        self.update(data)
        replay = np.array(self._replay)
        self._model.fit(replay)

    def predict(
        self, data: np.ndarray, *_args, **_kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "fogml-anomaly"

    def additional_info(self) -> Dict:
        simple_threesum_result = simple_threesum(1, 2, 3)
        return {"model": self._model.name(), "buffer_size": len(self._replay), "simple_threesum": simple_threesum_result}

