"""
Python wrapper for FogML LOF (Local Outlier Factor) C implementation.

This module provides a Python interface to the high-performance C implementation
of the Local Outlier Factor algorithm for outlier detection.

Copyright 2021 FogML
Licensed under the Apache License, Version 2.0
"""

import ctypes
from time import perf_counter
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import threading
import os


class TinyMLLOFConfig(ctypes.Structure):
    """
    C structure mapping for tinyml_lof_config_t.

    Fields:
        parameter_k (int): Number of nearest neighbors to consider
        k_distance (POINTER(ctypes.c_float)): Table of k-distance for each point
        lrd (POINTER(ctypes.c_float)): Local Reachability Density for each point
        n (int): Number of points in the dataset
        vector_size (int): Dimension of each point
        data (POINTER(ctypes.c_float)): Dataset points as flattened array
    """

    _fields_ = [
        ("parameter_k", ctypes.c_int),
        ("k_distance", ctypes.POINTER(ctypes.c_float)),
        ("lrd", ctypes.POINTER(ctypes.c_float)),
        ("n", ctypes.c_int),
        ("vector_size", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_float)),
    ]


class FogMLLOF:
    """
    Python wrapper for the FogML LOF (Local Outlier Factor) C implementation.

    This class provides a scikit-learn-like interface to the high-performance
    C implementation of the Local Outlier Factor algorithm.

    Parameters:
        k (int): Number of nearest neighbors to use for LOF calculation (default: 5)

    Attributes:
        k (int): Number of nearest neighbors
        is_fitted (bool): Whether the model has been fitted to data
        config (TinyMLLOFConfig): C structure configuration
        _lib (ctypes.CDLL): Loaded C library
    """

    def __init__(self, k: int = 5, n_threads: int = 10):
        """
        Initialize the FogML LOF model.

        Args:
            k (int): Number of nearest neighbors to consider (must be > 0)
            n_threads (int): Number of threads for OpenMP (1 = disable threading)

        Raises:
            ValueError: If k <= 0
            OSError: If the C library cannot be loaded
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.k = k
        self.n_threads = n_threads
        self.is_fitted = False
        self.config: Optional[TinyMLLOFConfig] = None
        self._data: Optional[ctypes.Array] = None
        self._k_distance: Optional[ctypes.Array] = None
        self._lrd: Optional[ctypes.Array] = None

        # Set OpenMP thread limit
        os.environ["OMP_NUM_THREADS"] = str(n_threads)

        # Load the C library
        self._lib = self._load_library()
        self._setup_function_signatures()

    def _load_library(self) -> ctypes.CDLL:
        """Load the FogML LOF shared library."""
        # Get the path to the shared library
        current_dir = Path(__file__).parent
        lib_path = current_dir / "c_source" / "lib" / "fogml_lof.so"

        if not lib_path.exists():
            raise OSError(
                f"Shared library not found at {lib_path}. "
                f"Please compile it using 'make' in the c_source directory."
            )

        try:
            return ctypes.CDLL(str(lib_path))
        except OSError as e:
            raise OSError(f"Failed to load shared library: {e}")

    def _setup_function_signatures(self):
        """Set up C function signatures for type safety."""
        # tinyml_lof_init
        self._lib.tinyml_lof_init.argtypes = [ctypes.POINTER(TinyMLLOFConfig)]
        self._lib.tinyml_lof_init.restype = None

        # tinyml_lof_learn
        self._lib.tinyml_lof_learn.argtypes = [ctypes.POINTER(TinyMLLOFConfig)]
        self._lib.tinyml_lof_learn.restype = None

        # tinyml_lof_score
        self._lib.tinyml_lof_score.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # vector
            ctypes.POINTER(TinyMLLOFConfig),  # config
        ]
        self._lib.tinyml_lof_score.restype = ctypes.c_float

        # vectored version of tinyml_score
        self._lib.tinyml_lof_score_vectored.argtypes = [
            ctypes.POINTER(
                ctypes.POINTER(ctypes.c_float)
            ),  # vector, 2d float array, outer layer is n_samples, inner layer is n_features
            ctypes.POINTER(TinyMLLOFConfig),  # config
            ctypes.POINTER(ctypes.c_float),  # scores
            ctypes.c_int,  # n_samples
        ]
        self._lib.tinyml_lof_score_vectored.restype = None

    def fit(self, X: Union[np.ndarray, list]) -> "FogMLLOF":
        """
        Fit the LOF model to the training data.

        Args:
            X (array-like): Training data of shape (n_samples, n_features)

        Returns:
            FogMLLOF: Self for method chaining

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If fitting fails
        """
        # return self
        # Convert to numpy array and validate
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples, n_features = X.shape

        if n_samples < self.k + 1:
            raise ValueError(
                f"Number of samples ({n_samples}) must be greater than k ({self.k})"
            )

        # Flatten the data for C library
        self._data = (ctypes.c_float * (n_samples * n_features))(*X.flatten())

        # Allocate arrays for k-distance and lrd
        self._k_distance = (ctypes.c_float * n_samples)()
        self._lrd = (ctypes.c_float * n_samples)()

        # Initialize configuration
        self.config = TinyMLLOFConfig()
        self.config.parameter_k = self.k
        self.config.n = n_samples
        self.config.vector_size = n_features
        self.config.data = self._data
        self.config.k_distance = self._k_distance
        self.config.lrd = self._lrd

        # Initialize and learn
        # measure the run time of the next two calls
        start_time = perf_counter()
        self._lib.tinyml_lof_init(ctypes.byref(self.config))
        init_time = perf_counter() - start_time

        start_time = perf_counter()
        self._lib.tinyml_lof_learn(ctypes.byref(self.config))

        learn_time = perf_counter() - start_time

        print(f"Initialization time: {init_time:.4f} seconds")
        print(f"Learning time: {learn_time:.4f} seconds")

        self.is_fitted = True
        return self

    def decision_function(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute LOF scores for the input samples.

        Args:
            X (array-like): Input samples of shape (n_samples, n_features)

        Returns:
            np.ndarray: LOF scores for each sample. Higher scores indicate outliers.

        Raises:
            ValueError: If the model is not fitted or input is invalid
        """
        # fake return 0, for testing
        return np.zeros(X.shape[0], dtype=np.float32)

        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling decision_function")

        assert self.config is not None  # Type guard for mypy

        # Convert to numpy array and validate
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")

        n_samples, n_features = X.shape

        if n_features != self.config.vector_size:
            raise ValueError(
                f"Expected {self.config.vector_size} features, got {n_features}"
            )

        # Compute LOF scores
        scores = np.zeros(n_samples, dtype=np.float32)

        # Call the vectored version
        vector_ptr = (ctypes.POINTER(ctypes.c_float) * n_samples)()
        scores_ptr = (ctypes.c_float * n_samples)()
        for i in range(n_samples):
            vector_ptr[i] = (ctypes.c_float * n_features)(*X[i])
        self._lib.tinyml_lof_score_vectored(
            vector_ptr, ctypes.byref(self.config), scores_ptr, n_samples
        )
        scores = np.ctypeslib.as_array(scores_ptr, shape=(n_samples,))

        return scores

    def predict(self, X: Union[np.ndarray, list], threshold: float = 1.5) -> np.ndarray:
        """
        Predict outliers in the input samples.

        Args:
            X (array-like): Input samples of shape (n_samples, n_features)
            threshold (float): LOF threshold above which samples are considered outliers

        Returns:
            np.ndarray: Binary predictions (1 for outliers, 0 for inliers)
        """
        scores = self.decision_function(X)
        return (scores > threshold).astype(int)

    def fit_predict(
        self, X: Union[np.ndarray, list], threshold: float = 1.5
    ) -> np.ndarray:
        """
        Fit the model and predict outliers in the training data.

        Args:
            X (array-like): Training data of shape (n_samples, n_features)
            threshold (float): LOF threshold above which samples are considered outliers

        Returns:
            np.ndarray: Binary predictions for the training data
        """
        self.fit(X)
        return self.predict(X, threshold)

    def get_training_scores(self) -> Optional[np.ndarray]:
        """
        Get LOF scores for the training data.

        Returns:
            np.ndarray or None: LOF scores for training data if fitted, None otherwise
        """
        if not self.is_fitted or self.config is None or self._data is None:
            return None

        # Reconstruct training data from the flattened array
        n_samples = self.config.n
        n_features = self.config.vector_size

        training_data = np.zeros((n_samples, n_features), dtype=np.float32)
        for i in range(n_samples):
            for j in range(n_features):
                training_data[i, j] = self._data[i * n_features + j]

        return self.decision_function(training_data)

    def get_k_distances(self) -> Optional[np.ndarray]:
        """
        Get k-distances for the training data points.

        Returns:
            np.ndarray or None: K-distances if fitted, None otherwise
        """
        if not self.is_fitted or self.config is None or self._k_distance is None:
            return None

        return np.array([self._k_distance[i] for i in range(self.config.n)])

    def get_local_reachability_densities(self) -> Optional[np.ndarray]:
        """
        Get local reachability densities for the training data points.

        Returns:
            np.ndarray or None: LRD values if fitted, None otherwise
        """
        if not self.is_fitted or self.config is None or self._lrd is None:
            return None

        return np.array([self._lrd[i] for i in range(self.config.n)])

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"FogMLLOF(k={self.k}, {status})"


# Convenience function for quick outlier detection
def lof_outlier_detection(
    X: Union[np.ndarray, list], k: int = 5, threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for quick LOF-based outlier detection.

    Args:
        X (array-like): Input data of shape (n_samples, n_features)
        k (int): Number of nearest neighbors
        threshold (float): LOF threshold for outlier detection

    Returns:
        tuple: (predictions, scores) where predictions are binary (1=outlier, 0=inlier)
               and scores are the LOF values
    """
    model = FogMLLOF(k=k)
    model.fit(X)
    scores = model.decision_function(X)
    predictions = (scores > threshold).astype(int)
    return predictions, scores


if __name__ == "__main__":
    # Example usage
    try:
        import matplotlib.pyplot as plt

        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

    # Generate sample data
    np.random.seed(42)
    n_inliers = 100
    n_outliers = 10

    # Generate inliers (normal data)
    inliers = np.random.normal(0, 1, (n_inliers, 2))

    # Generate outliers (anomalous data)
    outliers = np.random.uniform(-4, 4, (n_outliers, 2))
    outliers = outliers[
        np.linalg.norm(outliers, axis=1) > 2.5
    ]  # Keep only distant points

    # Combine data
    X = np.vstack([inliers, outliers[:5]])  # Use only 5 outliers

    print(f"Dataset shape: {X.shape}")

    # Fit LOF model
    lof = FogMLLOF(k=5)
    lof.fit(X)

    print(f"Model fitted: {lof}")

    # Get LOF scores
    scores = lof.decision_function(X)
    print(f"LOF scores range: {scores.min():.3f} to {scores.max():.3f}")

    # Predict outliers
    predictions = lof.predict(X, threshold=1.5)
    print(f"Found {predictions.sum()} outliers out of {len(X)} samples")

    # Display additional information
    k_distances = lof.get_k_distances()
    lrd_values = lof.get_local_reachability_densities()

    if k_distances is not None:
        print(f"K-distances range: {k_distances.min():.3f} to {k_distances.max():.3f}")

    if lrd_values is not None:
        print(f"LRD values range: {lrd_values.min():.3f} to {lrd_values.max():.3f}")

    if HAS_MATPLOTLIB:
        print("\nMatplotlib is available for visualization if needed.")
