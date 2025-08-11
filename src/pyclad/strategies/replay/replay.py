from typing import Dict

import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptAwareStrategy,
    ConceptIncrementalStrategy,
)


class ReplayOnlyStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    def __init__(self, model: Model, buffer: ReplayBuffer):
        self._model = model
        self._buffer = buffer

    def learn(self, data: np.ndarray, **kwargs) -> None:
        self._buffer.update(data)
        self._model.fit(self._buffer.data())

    def predict(self, data: np.ndarray, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "ReplayOnly"

    def additional_info(self) -> Dict:
        return {"replay_buffer": self._buffer.info()}


class ReplayEnhancedStrategy(
    ConceptAgnosticStrategy, ConceptIncrementalStrategy, ConceptAwareStrategy
):
    def __init__(self, model: Model, buffer: ReplayBuffer):
        self._model = model
        self._buffer = buffer

    def learn(self, data: np.ndarray, **kwargs) -> None:
        combined_data = (
            np.concatenate([self._buffer.data(), data])
            if len(self._buffer.data()) > 0
            else data
        )
        print(f"Fitting model on combined data of shape: {combined_data.shape}")
        self._model.fit(combined_data)
        self._buffer.update(data)

    def predict(self, data: np.ndarray, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "ReplayEnhanced"

    def additional_info(self) -> Dict:
        return {"replay_buffer": self._buffer.info()}
