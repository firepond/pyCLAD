"""
WATCH-based Reservoir Sampling Strategy for Lifelong Anomaly Detection.

This module implements a concept-aware strategy that uses WATCH (Windowed Adaptive
Test for Change) to detect regime changes and maintain a balanced reservoir of samples
from different regimes to prevent catastrophic forgetting.
"""

import logging
import math
from typing import Dict, Tuple

import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptAwareStrategy,
    ConceptIncrementalStrategy,
)

logger = logging.getLogger(__name__)


class Regime:
    """
    Represents a regime (concept) in the data stream.

    A regime encapsulates a set of samples that share similar characteristics
    and maintains statistics about the samples it contains.
    """

    def __init__(self, samples: np.ndarray):
        """
        Initialize a regime with initial samples.

        Args:
            samples (np.ndarray): Initial samples for the regime, must be 2D array.

        Raises:
            ValueError: If samples is not a 2D array.
        """
        if samples.ndim != 2:
            raise ValueError("Samples must be a 2D array.")

        self.r_samples = samples
        self.r_mean = samples.mean(axis=0)  # Mean of the current regime
        self.r_count = samples.shape[0]  # Number of samples in the current regime

    def __repr__(self) -> str:
        return f"Regime(mean={self.r_mean}, count={self.r_count})"

    def __len__(self) -> int:
        return self.r_count

    def __getitem__(self, item):
        return self.r_samples[item]

    def get_samples(self) -> np.ndarray:
        """Return the samples of the regime."""
        return self.r_samples

    def add_samples(self, new_samples: np.ndarray) -> None:
        """
        Add new samples to the regime.

        Args:
            new_samples (np.ndarray): New samples to add, must be 2D array.

        Raises:
            ValueError: If new_samples is not a 2D array.
        """
        if new_samples.ndim != 2:
            raise ValueError("New samples must be a 2D array.")

        self.r_samples = np.vstack([self.r_samples, new_samples])
        self.r_count = self.r_samples.shape[0]
        self.r_mean = self.r_samples.mean(axis=0)

    def get_mean(self) -> np.ndarray:
        """Return the mean of the regime."""
        return self.r_mean


def chebyshev_min(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the minimum Chebyshev distance between two arrays.

    Args:
        u (np.ndarray): First array.
        v (np.ndarray): Second array.

    Returns:
        float: Minimum Chebyshev distance.
    """
    return np.min(np.abs(u - v))


class WatchStrategy(
    ConceptAwareStrategy, ConceptIncrementalStrategy, ConceptAgnosticStrategy
):
    """
    WATCH-based reservoir sampling strategy for lifelong anomaly detection.

    This strategy implements the core of lifelong learning by using microWATCH
    to check if new batches represent new regimes. If so, a new regime is added
    to the reservoir. If not, the best-matching existing regime is updated.
    This prevents catastrophic forgetting by maintaining a balanced view of past data.
    """

    def __init__(
        self, model: Model, max_buffer_size: int = 1000, threshold_ratio: float = 0.51
    ):
        """
        Initialize the WATCH strategy.

        Args:
            model (Model): The anomaly detection model to use.
            max_buffer_size (int): Maximum size of the replay buffer.
            threshold_ratio (float): Threshold for regime similarity comparison.
        """
        self._replay = []  # List of replay regimes
        self._model = model
        self.max_buffer_size = max_buffer_size
        self.current_size = 0
        self.threshold = threshold_ratio

    def _calculate_distance(self, regime1: Regime, regime2: Regime) -> float:
        """
        Calculate distance between two regimes.

        Args:
            regime1 (Regime): First regime.
            regime2 (Regime): Second regime.

        Returns:
            float: Distance between the regimes.
        """
        distance = chebyshev_min(regime1.get_samples(), regime2.get_mean())
        # Alternative: Mahalanobis distance
        # distance = mahalanobis_batch_to_dist(regime1.get_samples(), regime2.get_samples())
        return distance

    def _find_best_matching_regime(self, new_regime: Regime) -> Tuple[float, int]:
        """
        Find the best matching regime for a new regime.

        Args:
            new_regime (Regime): The new regime to match.

        Returns:
            Tuple[float, int]: Best match distance and index of the best matching regime.
        """
        if not self._replay:
            return float("inf"), -1

        best_distance = float("inf")
        best_index = -1

        for i, regime in enumerate(self._replay):
            distance = self._calculate_distance(new_regime, regime)
            if distance < best_distance:
                best_distance = distance
                best_index = i

        return best_distance, best_index

    def _should_create_new_regime(self, distance: float) -> bool:
        """
        Determine if a new regime should be created based on distance.

        Args:
            distance (float): Distance to the closest existing regime.

        Returns:
            bool: True if a new regime should be created.
        """
        return distance > self.threshold

    def _update_regime_size_limits(self) -> None:
        """Update regime sizes to fit within the buffer limit."""
        if self.current_size <= self.max_buffer_size:
            return

        # Equal proportions for all regimes
        buffer_limit = math.ceil(self.max_buffer_size / len(self._replay))

        for regime in self._replay:
            if len(regime) > buffer_limit:
                # Sample from the regime to reduce its size
                regime.r_samples = self._select_samples(regime.r_samples, buffer_limit)
                regime.r_count = buffer_limit

        # Update current size after sampling
        self.current_size = sum(regime.r_count for regime in self._replay)

        # Remove empty regimes
        self._replay = [regime for regime in self._replay if regime.r_count > 0]

    def _select_samples(self, buffer: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Randomly select n_samples from the buffer.

        Args:
            buffer (np.ndarray): Buffer to sample from.
            n_samples (int): Number of samples to select.

        Returns:
            np.ndarray: Selected samples.
        """
        if n_samples >= buffer.shape[0]:
            return buffer

        indices = np.random.choice(buffer.shape[0], n_samples, replace=False)
        return buffer[indices]

    def update(self, data: np.ndarray) -> None:
        """
        Update the replay regimes with new data.

        Args:
            data (np.ndarray): New data batch to process.
        """
        new_regime = Regime(data)

        if not self._replay:
            # If there are no regimes, add the new regime
            self._replay.append(new_regime)
            self.current_size += data.shape[0]
            return

        # Find the best matching regime
        best_distance, best_index = self._find_best_matching_regime(new_regime)

        if self._should_create_new_regime(best_distance):
            # Create a new regime if the distance is above threshold
            self._replay.append(new_regime)
            logger.debug("New regime added: %s", new_regime)
        else:
            # Update the existing regime with the new data
            self._replay[best_index].add_samples(new_regime.get_samples())
            logger.debug("Regime updated: %s", self._replay[best_index])

        # Update current size
        self.current_size = sum(regime.r_count for regime in self._replay)

        # Manage buffer size limits
        self._update_regime_size_limits()

    def learn(self, data: np.ndarray, *_args, **_kwargs) -> None:
        """
        Learn from the data and update the model.

        Args:
            data (np.ndarray): New data to learn from.
            *_args: Additional positional arguments (unused but required for interface compatibility).
            **_kwargs: Additional keyword arguments (unused but required for interface compatibility).
        """
        # Collect all regime samples
        replay_samples = [regime.r_samples for regime in self._replay]

        if replay_samples:
            replay_samples.append(data)
            combined_data = np.concatenate(replay_samples)
        else:
            combined_data = data

        # Fit the model on the combined data
        self._model.fit(combined_data)

        # Update the replay buffer
        self.update(data)

    def predict(
        self, data: np.ndarray, *_args, **_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on the data.

        Args:
            data (np.ndarray): Data to make predictions on.
            *_args: Additional positional arguments (unused but required for interface compatibility).
            **_kwargs: Additional keyword arguments (unused but required for interface compatibility).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and anomaly scores.
        """
        return self._model.predict(data)

    def name(self) -> str:
        """Return the name of the strategy."""
        return "BalancedWATCHReservoirSampling"

    def additional_info(self) -> Dict:
        """
        Return additional information about the strategy.

        Returns:
            Dict: Additional information including strategy type and buffer size.
        """
        return {
            "model": self._model.name(),
            "strategy": "WATCH",
            "buffer_size": self.max_buffer_size,
            "current_size": self.current_size,
            "num_regimes": len(self._replay),
            "threshold": self.threshold,
        }
