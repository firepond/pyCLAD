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


def manhattan(u: np.ndarray, v: np.ndarray) -> float:
    return np.sum(np.abs(u - v))


def acc(u, v):
    # the acc distance
    return (manhattan(u, v) + chebyshev_min(u, v)) / 2


def euclidean(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two arrays.

    Args:
        u (np.ndarray): First array.
        v (np.ndarray): Second array.

    Returns:
        float: Euclidean distance.
    """
    return np.linalg.norm(u - v)


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
        self, model: Model, max_buffer_size: int = 1000, threshold_ratio: float = 0.1
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
        self.threshold_ratio = threshold_ratio  # multiplier for the threshold ratio

        self.cur_threshold = threshold_ratio

        self.past_distances = []
        # best matching distances of all past regimes,
        # used to calculate the current threshold,
        # so 90 percentile of the past distances are greater than the current threshold

        self.mean = None  # mean of all past regimes
        self.sum = None  # sum of all past regimes
        self.count = 0  # count of all past regimes

    def _calculate_distance(self, regime1: Regime, regime2: Regime) -> float:
        """
        Calculate distance between two regimes.

        Args:
            regime1 (Regime): First regime.
            regime2 (Regime): Second regime.

        Returns:
            float: Distance between the regimes.
        """
        # distance = chebyshev_min(regime1.get_samples(), regime2.get_mean())
        # distance = acc(regime1.get_samples(), regime2.get_mean())
        # Use Euclidean distance for regime comparison
        distance = euclidean(regime1.get_mean(), regime2.get_mean())
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
        return distance > self.cur_threshold

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
            # print("Initial regime added")
            return

        # Find the best matching regime
        best_distance, best_index = self._find_best_matching_regime(new_regime)
        self.past_distances.append(best_distance)

        # Calculate the current threshold based on past distances
        if len(self.past_distances) > 3:
            # Use the 90th percentile of past distances to set the threshold
            percent = np.percentile(self.past_distances, 90)
            # if percent is not a scalar, take the first element
            if isinstance(percent, np.ndarray):
                print(
                    "Percentile is not a scalar, taking the first element. Percentile shape:",
                    percent.shape,
                )
                percent = percent[0]
            if percent == 0:
                # self.cur_threshold = 0%.6f", self.cur_threshold)
                pass
            else:
                self.cur_threshold = percent * self.threshold_ratio

        # print cur threshold
        # print(f"Current threshold: {self.cur_threshold:.6f}")
        # should create a new regime if the distance is above the threshold or only 2 regimes exist
        if self._should_create_new_regime(best_distance) or len(self._replay) < 2:
            # Create a new regime if the distance is above the threshold
            self._replay.append(new_regime)
            # print("New regime added")
        else:
            # Update the existing regime with the new data
            self._replay[best_index].add_samples(new_regime.get_samples())
            # print("Regime updated")

        # Update current size
        # self.current_size = sum(regime.r_count for regime in self._replay)

        # Manage buffer size limits
        self._update_regime_size_limits()
        # update the mean, sum and count for the threshold calculation
        # self.sum = (
        #     np.sum(data, axis=0)
        #     if self.sum is None
        #     else self.sum + np.sum(data, axis=0)
        # )
        # self.count += data.shape[0]
        # self.mean = self.sum / self.count if self.count > 0 else 0
        # # assert  self.mean is ndarray
        # if type(self.mean) is not np.ndarray:
        #     self.mean = np.array(self.mean)
        # dist = chebyshev_min(self.mean, new_regime.get_mean())
        # if dist > self.max_distance:
        #     self.max_distance = dist
        #     print("New maximum distance found: %.2f", self.max_distance)
        # # Update the current threshold based on the maximum distance
        # if self.max_distance > 0:
        #     self.cur_threshold = self.threshold_ratio * self.max_distance

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
            "threshold": self.threshold_ratio,
        }
