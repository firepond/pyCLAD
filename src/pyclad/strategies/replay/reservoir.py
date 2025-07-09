import logging
import math
from typing import Dict

import numpy as np
from scipy.stats import entropy

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptAwareStrategy,
    ConceptIncrementalStrategy,
)

logger = logging.getLogger(__name__)


class ReservoirSamplingStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy):
    def __init__(self, model: Model, max_buffer_size: int = 1000):
        self._replay = []
        self._model = model
        self.max_buffer_size = max_buffer_size
        self.buffer_size = 0

    def update(self, data: np.ndarray):
        """Update the replay buffer with new data."""
        # use reservoir sampling to keep the buffer size manageable
        for sample in data:
            if self.buffer_size < self.max_buffer_size:
                self._replay.append(sample)
                self.buffer_size += 1
            else:
                # replace a random sample in the buffer with the new sample
                index = np.random.randint(0, self.max_buffer_size)
                self._replay[index] = sample

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        """Learn from the data and store it in the replay buffer."""
        self.update(data)
        # concatenate the replay buffer and fit the model
        if len(self._replay) == 0:
            logger.warning("Replay buffer is empty. No data to fit the model.")
            return
        replay = np.array(self._replay)
        # fit the model on the concatenated replay buffer
        self._model.fit(replay)

    def predict(self, data: np.ndarray, *args, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "ReservoirSampling"

    def additional_info(self) -> Dict:
        return {"model": self._model.name(), "buffer_size": len(np.concatenate(self._replay))}


def calculate_lof_entropy(lof_scores, bins=20):
    """
    Calculates the entropy of a set of LOF scores.

    Args:
        lof_scores (np.ndarray): An array of LOF scores from a fitted model.
        bins (int): The number of bins to use for discretizing the scores.
                    This is a crucial parameter that can affect the result.

    Returns:
        float: The calculated entropy value in nats.
    """
    # 1. Create a histogram of the LOF scores. This gives us the count
    #    of scores falling into each bin (a discrete distribution).
    counts, bin_edges = np.histogram(lof_scores, bins=bins, density=False)

    # 2. Convert the counts into probabilities by dividing by the total number of scores.
    #    This gives us the probability distribution p(x).
    probabilities = counts / len(lof_scores)

    # 3. Calculate the Shannon entropy using scipy.stats.entropy.
    #    This function computes H(X) = -sum(p(x) * log(p(x))).
    #    It automatically handles bins with zero probability.
    return entropy(probabilities, base=np.e)


class BalancedReservoirSamplingStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy):
    # lifelong anomaly detection strategy that uses reservoir sampling
    def __init__(self, model: Model, max_buffer_size: int = 1000):
        self._replay = []  # list to hold the replay buffers, one buffer for each concept
        # merge the replay buffers when learning

        # anomaly scores for the replay buffers, calculate the complexity of each buffer at the first update, then use the scores to select samples
        self._replay_remove_scores = []

        self._model = model
        self.max_buffer_size = max_buffer_size
        self.current_size = 0

    def update(self, data: np.ndarray):
        """Update the replay buffer with new data."""
        self._replay.append(data)

        # reservoir sampling to keep the buffer size manageable
        # set a maximum size for each concept's buffer, then randomly sample from the buffers
        self.current_size += data.shape[0]
        # calculate the anomaly scores for the newest data
        if len(self._replay) > 0:
            self._replay_remove_scores.append(self._model.predict(data)[1])

        if self.current_size > self.max_buffer_size:
            # calculate complexity for each concept's buffer, based on the variance and entropy
            complexity = []
            for buffer in self._replay:
                if buffer.shape[0] > 0:
                    # complexity_value = np.var(buffer)
                    var_value = np.var(buffer)
                    entropy_value = calculate_lof_entropy(buffer)
                    # calculate the complexity as a combination of variance and entropy, mean
                    complexity_value = var_value
                    complexity.append(complexity_value)

            summed_complexity = sum(complexity)
            if summed_complexity > 0:
                proportions = [c / summed_complexity for c in complexity]
            else:
                proportions = [1 / len(complexity)] * len(complexity)

            proportions = [1 / len(self._replay)] * len(self._replay)  # equal proportions for all buffers

            for i in range(len(self._replay)):
                cur_buffer_limit = math.ceil(self.max_buffer_size * proportions[i])
                if self._replay[i].shape[0] > cur_buffer_limit:
                    # sample from the buffer to reduce its size
                    # self._replay[i] = self.select_samples(self._replay[i], cur_buffer_limit)
                    # self._replay[i] = self.select_samples_lof_score(self._replay[i], cur_buffer_limit)  # 0.808
                    # self._replay[i] = self.select_samples_lof_probability(self._replay[i], cur_buffer_limit)  # 0.7459
                    self._replay[i] = self.select_samples_lof_density(self._replay[i], cur_buffer_limit)  # 0.730
                    # self._replay[i] = self.select_samples_lof_score_static(i, cur_buffer_limit)

        # update the current size after sampling
        self.current_size = sum(buffer.shape[0] for buffer in self._replay)

        # remove empty buffers
        self._replay = [buffer for buffer in self._replay if buffer.shape[0] > 0]

    def select_samples(self, buffer: np.ndarray, n_samples: int) -> np.ndarray:
        """Randomly select n_samples from the replay buffer"""
        shape = buffer.shape
        result_shape = (n_samples,) + shape[1:]
        if n_samples >= shape[0]:
            return buffer
        indices = np.random.choice(shape[0], n_samples, replace=False)
        result = buffer[indices].reshape(result_shape)
        return result

    def select_samples_lof_score(self, buffer: np.ndarray, n_samples: int) -> np.ndarray:
        """Select samples based on Local Outlier Factor (LOF) scores."""
        lof_scores = self._model.predict(buffer)[1]
        sorted_indices = np.argsort(lof_scores)
        # select the top n_samples with the lowest LOF scores
        selected_indices = sorted_indices[:n_samples]
        return buffer[selected_indices]

    def select_samples_lof_score_static(self, index: int, n_samples: int) -> np.ndarray:
        """Select samples based on Local Outlier Factor (LOF) scores, based on the static remove scores."""
        if index >= len(self._replay_remove_scores):
            raise IndexError("Index out of bounds for replay remove scores.")
        lof_scores = self._replay_remove_scores[index]
        sorted_indices = np.argsort(lof_scores)
        # select the top n_samples with the lowest LOF scores
        selected_indices = sorted_indices[:n_samples]
        # also only keep the selected indices in the remove scores
        self._replay_remove_scores[index] = self._replay_remove_scores[index][selected_indices]
        return self._replay[index][selected_indices]

    def select_samples_lof_probability(self, buffer: np.ndarray, n_samples: int) -> np.ndarray:
        """Select samples based on Local Outlier Factor (LOF) scores as probabilities."""
        lof_scores = self._model.predict(buffer)[1]
        inverse_lof = 1 / lof_scores

        lof_scores = np.nan_to_num(inverse_lof, nan=0.0, posinf=0.0, neginf=0.0)
        # normalize the LOF scores to get probabilities
        probabilities = (lof_scores - np.min(lof_scores)) / (np.max(lof_scores) - np.min(lof_scores))
        probabilities /= np.sum(probabilities)  # normalize
        # select samples based on the probabilities
        selected_indices = np.random.choice(np.arange(buffer.shape[0]), size=n_samples, replace=False, p=probabilities)
        return buffer[selected_indices]

    def select_samples_lof_density(self, buffer: np.ndarray, n_samples: int) -> np.ndarray:
        """Select samples based on Local Outlier Factor (LOF) density."""
        if n_samples >= buffer.shape[0]:
            return buffer

        # 1. Get LOF scores
        lof_scores = -self._model.predict(buffer)[1]  # Invert scores so higher is more outlier-like

        # 2. Define thresholds
        inlier_threshold = np.percentile(lof_scores, 80)
        outlier_threshold = np.percentile(lof_scores, 98)

        # 3. Categorize points
        indices_inlier = np.where(lof_scores <= inlier_threshold)[0]
        indices_borderline = np.where((lof_scores > inlier_threshold) & (lof_scores <= outlier_threshold))[0]
        indices_outlier = np.where(lof_scores > outlier_threshold)[0]

        # 4. Determine subsample composition
        prop_inlier = len(indices_inlier) / buffer.shape[0]
        prop_borderline = len(indices_borderline) / buffer.shape[0]

        n_inlier = int(n_samples * prop_inlier)
        n_borderline = int(n_samples * prop_borderline)
        n_outlier = n_samples - n_inlier - n_borderline

        # 5. Sample from each category
        subsample_indices_inlier = np.random.choice(indices_inlier, min(n_inlier, len(indices_inlier)), replace=False)
        subsample_indices_borderline = np.random.choice(
            indices_borderline, min(n_borderline, len(indices_borderline)), replace=False
        )
        subsample_indices_outlier = np.random.choice(
            indices_outlier, min(n_outlier, len(indices_outlier)), replace=False
        )

        # 6. Combine and return
        subsample_indices = np.concatenate(
            [subsample_indices_inlier, subsample_indices_borderline, subsample_indices_outlier]
        )

        # If we don't have enough samples, fill with random samples from the buffer
        if len(subsample_indices) < n_samples:
            remaining_needed = n_samples - len(subsample_indices)
            remaining_indices = np.setdiff1d(np.arange(buffer.shape[0]), subsample_indices)
            fill_indices = np.random.choice(remaining_indices, remaining_needed, replace=False)
            subsample_indices = np.concatenate([subsample_indices, fill_indices])

        return buffer[subsample_indices]

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        replay = np.concatenate(self._replay) if len(self._replay) > 0 else np.empty((0, data.shape[1]))

        self._model.fit(np.concatenate([replay, data]) if replay.shape[0] > 0 else data)
        self.update(data)

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def additional_info(self) -> Dict:
        return {"model": self._model.name(), "buffer_size": self.max_buffer_size}

    def name(self) -> str:
        return "BalancedReservoirSampling"
