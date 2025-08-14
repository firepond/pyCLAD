"""
FogML LOF Model implementation for pyCLAD.

This module provides a Model-compliant wrapper around the high-performance
C implementation of the Local Outlier Factor algorithm from FogML.
"""

import numpy as np
from typing import Tuple, Any, Dict

from ..model import Model
from .fogml_lof_wrapper import FogMLLOF


class FogMLLOFModel(Model):
    """
    FogML LOF Model wrapper for pyCLAD.

    This class wraps the high-performance C implementation of LOF to provide
    a unified interface compatible with the pyCLAD Model base class.

    Parameters:
        k (int): Number of nearest neighbors to consider (default: 5)
        threshold (float): LOF threshold for outlier classification (default: 1.5)
    """

    def __init__(self, k: int = 5, threshold: float = 1.5):
        """
        Initialize the FogML LOF model.

        Args:
            k (int): Number of nearest neighbors to consider (must be > 0)
            threshold (float): LOF threshold above which samples are considered outliers

        Raises:
            ValueError: If k <= 0
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.k = k
        self.threshold = threshold
        self._model = FogMLLOF(k=k)
        self._is_fitted = False

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the LOF model to the training data.

        Args:
            data (np.ndarray): Training data of shape (n_samples, n_features)

        Raises:
            ValueError: If input data is invalid
        """
        self._model.fit(data)
        self._is_fitted = True

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict outliers and compute anomaly scores for the input samples.

        Args:
            data (np.ndarray): Input samples of shape (n_samples, n_features)

        Returns:
            tuple: (labels, scores) where:
                - labels (np.ndarray): Binary predictions (0 for normal, 1 for anomaly)
                - scores (np.ndarray): Anomaly scores (higher values = more anomalous)

        Raises:
            ValueError: If the model is not fitted or input is invalid
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before calling predict")

        # Get LOF scores from the underlying model
        lof_scores = self._model.decision_function(data)

        # Convert LOF scores to labels using threshold
        # LOF scores > threshold are outliers (label = 1)
        labels = (lof_scores > self.threshold).astype(int)

        # Use LOF scores directly as anomaly scores
        # Higher LOF scores indicate more anomalous samples
        scores = lof_scores

        return labels, scores

    def name(self) -> str:
        """Return the model name."""
        return "FogMLLOFModel"

    def additional_info(self) -> Dict[str, Any]:
        """
        Return additional model information.

        Returns:
            dict: Dictionary containing model parameters and state information
        """
        info = {
            "k_neighbors": self.k,
            "threshold": self.threshold,
            "is_fitted": self._is_fitted,
        }

        # Add training data statistics if the model is fitted
        if self._is_fitted:
            try:
                k_distances = self._model.get_k_distances()
                lrd_values = self._model.get_local_reachability_densities()

                if k_distances is not None:
                    info["k_distance_stats"] = {
                        "min": float(k_distances.min()),
                        "max": float(k_distances.max()),
                        "mean": float(k_distances.mean()),
                    }

                if lrd_values is not None:
                    info["lrd_stats"] = {
                        "min": float(lrd_values.min()),
                        "max": float(lrd_values.max()),
                        "mean": float(lrd_values.mean()),
                    }
            except Exception:
                # If getting internal stats fails, just continue without them
                pass

        return info

    def get_k_distances(self) -> np.ndarray:
        """
        Get k-distances for the training data points.

        Returns:
            np.ndarray: K-distances for each training sample

        Raises:
            ValueError: If the model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing k-distances")

        k_distances = self._model.get_k_distances()
        if k_distances is None:
            raise RuntimeError("Failed to retrieve k-distances from the model")

        return k_distances

    def get_local_reachability_densities(self) -> np.ndarray:
        """
        Get local reachability densities for the training data points.

        Returns:
            np.ndarray: LRD values for each training sample

        Raises:
            ValueError: If the model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing LRD values")

        lrd_values = self._model.get_local_reachability_densities()
        if lrd_values is None:
            raise RuntimeError("Failed to retrieve LRD values from the model")

        return lrd_values

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """
        Get raw LOF scores for the input samples.

        Args:
            data (np.ndarray): Input samples of shape (n_samples, n_features)

        Returns:
            np.ndarray: LOF scores (higher values = more anomalous)

        Raises:
            ValueError: If the model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before calling decision_function")

        return self._model.decision_function(data)

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return f"FogMLLOFModel(k={self.k}, threshold={self.threshold}, {status})"
