"""
CANDI (Continual ANomaly Detection for Iot): WATCH-based Reservoir Sampling Strategy for Lifelong Anomaly Detection.

This module implements a concept-aware strategy that uses WATCH to detect regime changes and maintain a balanced reservoir of samples
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

    def recalculate_mean(self):
        """Recalculates the mean from the samples."""
        if self.r_count > 0:
            self.r_mean = self.r_samples.mean(axis=0)
        else:
            # Handle case with no samples, perhaps by setting mean to an empty array or zero
            # Assuming r_samples has a defined dimension, even if empty
            if self.r_samples.ndim > 1 and self.r_samples.shape[1] > 0:
                self.r_mean = np.zeros(self.r_samples.shape[1])
            else:
                self.r_mean = np.array([])

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

        # Performance optimization: Use incremental mean calculation instead of recomputing
        old_count = self.r_count
        new_count = new_samples.shape[0]
        total_count = old_count + new_count

        # Only update mean if new_count > 0
        if new_count > 0:
            new_mean = new_samples.mean(axis=0)
            self.r_mean = (old_count * self.r_mean + new_count * new_mean) / total_count

            # Use np.concatenate for better performance if both arrays are contiguous
            self.r_samples = np.concatenate([self.r_samples, new_samples], axis=0)
            self.r_count = total_count

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


def calculate_iqr_threshold(history_of_distances, k=1.5, min_samples=2):
    """
    Calculates an adaptive threshold for outlier detection using the
    Interquartile Range (IQR) method (Tukey's Fences).

    This method is robust and well-suited for skewed distributions, which is
    common for distance metrics.

    Args:
        history_of_distances (np.ndarray): A 1D NumPy array containing the history
                                           of "best distances" from past normal
                                           observations.
        k (float, optional): The multiplier for the IQR. The standard value is 1.5
                             for detecting "mild outliers". Defaults to 1.5.
        min_samples (int, optional): The minimum number of historical distances
                                     required to calculate a stable threshold.
                                     Defaults to 2.

    Returns:
        float: The calculated adaptive threshold. Returns np.inf if there are
               not enough samples in the history, effectively disabling detection
               until enough data is gathered.
    """
    # --- Edge Case Handling ---
    # If the history is too small, we can't get a reliable statistical measure.
    # Return infinity to prevent any new mode creation during this "warm-up" period.
    if len(history_of_distances) < min_samples:
        return np.inf

    # --- Core Calculation ---
    # 1. Calculate the First Quartile (Q1 - the 25th percentile)
    q1 = np.percentile(history_of_distances, 25)

    # 2. Calculate the Third Quartile (Q3 - the 75th percentile)
    q3 = np.percentile(history_of_distances, 75)

    # 3. Calculate the Interquartile Range (IQR)
    iqr = q3 - q1

    # If IQR is zero (all historical distances are the same), the threshold is simply Q3
    if iqr == 0:
        # We add a tiny epsilon to ensure the threshold is slightly above the constant value
        return q3 + 1e-9

    # 4. Calculate the adaptive threshold (the "upper fence")
    threshold = q3 + k * iqr

    return threshold


def calculate_percentile_threshold(history_of_distances, percentile=90):
    """
    Calculate a percentile-based threshold for outlier detection.

    Args:
        history_of_distances (np.ndarray): A 1D NumPy array containing the history
                                           of "best distances" from past normal
                                           observations.
        percentile (float, optional): The percentile to use for the threshold.
                                       Defaults to 90.

    Returns:
        float: The calculated percentile threshold. Returns np.inf if there are
               not enough samples in the history, effectively disabling detection
               until enough data is gathered.
    """
    if len(history_of_distances) <= 2:
        return np.inf

    result = np.percentile(history_of_distances, percentile) * 0.2
    return result


def calculate_combined_threshold(history_of_distances, iqr_k=1.5, percentile=95):
    """
    Calculate a combined threshold for outlier detection using both IQR and percentile methods.

    Args:
        history_of_distances (np.ndarray): A 1D NumPy array containing the history
                                           of "best distances" from past normal
                                           observations.
        iqr_k (float, optional): The multiplier for the IQR. Defaults to 1.5.
        percentile (float, optional): The percentile to use for the threshold.
                                       Defaults to 95.

    Returns:
        float: The calculated combined threshold.
    """
    iqr_threshold = calculate_iqr_threshold(history_of_distances, k=iqr_k)
    percentile_threshold = calculate_percentile_threshold(
        history_of_distances, percentile=percentile
    )

    # Combine the thresholds using a simple average
    combined_threshold = min(iqr_threshold, percentile_threshold)

    return combined_threshold


class CandiStrategy(
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
        # distance = chebyshev_min(regime1.get_mean(), regime2.get_mean())
        # distance = acc(regime1.get_samples(), regime2.get_mean())
        # Use Euclidean distance for regime comparison
        distance = euclidean(regime1.get_mean(), regime2.get_mean())
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
        new_mean = new_regime.get_mean()
        # Use generator to avoid unnecessary array creation if only one regime
        if len(self._replay) == 1:
            dist = chebyshev_min(self._replay[0].get_mean(), new_mean)
            return dist, 0
        means = np.stack([regime.get_mean() for regime in self._replay])
        # Vectorized chebyshev_min: min(abs(mean - new_mean)) for each regime
        abs_diff = np.abs(means - new_mean)
        distances = np.min(abs_diff, axis=1)
        best_index = np.argmin(distances)
        best_distance = float(distances[best_index])
        return best_distance, int(best_index)

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
        print("Updating regime size limits.")
        """Update regime sizes to fit within the buffer limit."""

        # Equal proportions for all regimes
        buffer_limit = max(1, math.ceil(self.max_buffer_size / len(self._replay)))
        print(f"Buffer limit for each regime: {buffer_limit}")

        for regime in self._replay:
            if len(regime) > buffer_limit:
                # Sample from the regime to reduce its size
                regime.r_samples = self._select_samples(regime.r_samples, buffer_limit)
                regime.r_count = buffer_limit
                regime.recalculate_mean()  # Recalculate mean after resampling

        # Update current size after sampling
        self.current_size = sum(regime.r_count for regime in self._replay)

        # Remove empty regimes efficiently
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
        # Use np.random.Generator for better performance and reproducibility if needed
        rng = np.random.default_rng()
        indices = rng.choice(buffer.shape[0], n_samples, replace=False)
        return buffer[indices]

    def update(self, data: np.ndarray) -> None:
        """
        Update the replay regimes with new data.

        Args:
            data (np.ndarray): New data batch to process.
        """
        new_regime = Regime(data)
        past_distances = []

        if not self._replay:
            # If there are no regimes, add the new regime
            self._replay.append(new_regime)
            self.current_size += data.shape[0]
            # print("Initial regime added")

        # Find the best matching regime
        best_distance, best_index = self._find_best_matching_regime(new_regime)
        past_distances.append(best_distance)

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
            print(f"Updating regime {best_index} with new samples")
            self._replay[best_index].r_samples = (
                new_regime.get_samples()
            )  # try replacing

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

        # sub sample data to fit buffer size, the upper limit is self.max_buffer_size
        # randomly choose from data
        if self.resize_new_regime:
            if data.shape[0] > self.max_buffer_size:
                sub_samples = self._select_samples(data, self.max_buffer_size)
            else:
                sub_samples = data
        else:
            replay_samples = []

        if replay_samples:
            replay_samples.append(data)
            combined_data = np.concatenate(replay_samples)
        else:
            combined_data = data

        # Fit the model on the combined data
        print(f"Fitting model with {combined_data.shape[0]} samples")
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
