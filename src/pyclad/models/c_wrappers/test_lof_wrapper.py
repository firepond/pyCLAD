#!/usr/bin/env python3
"""
Test script for FogML LOF wrapper.
"""

import numpy as np
import sys
from pathlib import Path

# Add the wrapper to path
sys.path.insert(0, str(Path(__file__).parent))

from fogml_lof_wrapper import FogMLLOF, lof_outlier_detection


def test_basic_functionality():
    """Test basic LOF functionality."""
    print("=== Testing Basic Functionality ===")

    # Create simple test data
    np.random.seed(123)

    # Normal points clustered around origin
    normal_points = np.random.normal(0, 0.5, (20, 2))

    # Outlier points far from the cluster
    outlier_points = np.array([[3, 3], [-3, -3], [3, -3]])

    # Combine data
    X = np.vstack([normal_points, outlier_points])

    print(f"Dataset shape: {X.shape}")
    print(f"Normal points: {len(normal_points)}, Outliers: {len(outlier_points)}")

    # Initialize and fit LOF model
    lof = FogMLLOF(k=5)

    # Test model before fitting
    assert not lof.is_fitted
    assert lof.get_k_distances() is None
    assert lof.get_local_reachability_densities() is None

    # Fit the model
    lof.fit(X)
    assert lof.is_fitted
    print("✓ Model fitted successfully")

    # Test decision function
    scores = lof.decision_function(X)
    print(f"LOF scores shape: {scores.shape}")
    print(f"LOF scores range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test prediction
    predictions = lof.predict(X, threshold=1.5)
    n_outliers_detected = predictions.sum()
    print(f"Detected {n_outliers_detected} outliers")

    # Test additional methods
    k_distances = lof.get_k_distances()
    lrd_values = lof.get_local_reachability_densities()
    training_scores = lof.get_training_scores()

    assert k_distances is not None
    assert lrd_values is not None
    assert training_scores is not None

    print(f"✓ K-distances computed: shape {k_distances.shape}")
    print(f"✓ LRD values computed: shape {lrd_values.shape}")
    print(f"✓ Training scores computed: shape {training_scores.shape}")

    # Verify training scores match decision function
    np.testing.assert_array_almost_equal(scores, training_scores, decimal=5)
    print("✓ Training scores match decision function output")


def test_convenience_function():
    """Test the convenience function."""
    print("\n=== Testing Convenience Function ===")

    # Simple 2D data
    np.random.seed(456)
    X = np.random.normal(0, 1, (50, 3))
    X = np.vstack([X, [[5, 5, 5], [-5, -5, -5]]])  # Add clear outliers

    predictions, scores = lof_outlier_detection(X, k=5, threshold=1.5)

    print(f"Dataset shape: {X.shape}")
    print(f"Detected {predictions.sum()} outliers")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print("✓ Convenience function works correctly")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")

    # Test invalid k
    try:
        FogMLLOF(k=0)
        assert False, "Should raise ValueError for k=0"
    except ValueError:
        print("✓ Correctly rejects k=0")

    try:
        FogMLLOF(k=-1)
        assert False, "Should raise ValueError for negative k"
    except ValueError:
        print("✓ Correctly rejects negative k")

    # Test insufficient data
    lof = FogMLLOF(k=5)
    try:
        X_small = np.array([[1, 2], [3, 4]])  # Only 2 samples, k=5
        lof.fit(X_small)
        assert False, "Should raise ValueError for insufficient data"
    except ValueError:
        print("✓ Correctly rejects insufficient training data")

    # Test unfitted model
    lof = FogMLLOF(k=3)
    try:
        lof.decision_function([[1, 2]])
        assert False, "Should raise ValueError for unfitted model"
    except ValueError:
        print("✓ Correctly rejects prediction on unfitted model")

    # Test dimension mismatch
    lof = FogMLLOF(k=3)
    X_train = np.random.random((10, 2))
    lof.fit(X_train)

    try:
        X_test = np.random.random((5, 3))  # Different dimensions
        lof.decision_function(X_test)
        assert False, "Should raise ValueError for dimension mismatch"
    except ValueError:
        print("✓ Correctly rejects dimension mismatch")


def test_single_sample_prediction():
    """Test prediction on single samples."""
    print("\n=== Testing Single Sample Prediction ===")

    # Train on multiple samples
    np.random.seed(789)
    X_train = np.random.normal(0, 1, (20, 2))

    lof = FogMLLOF(k=5)
    lof.fit(X_train)

    # Test single sample (1D array)
    single_sample = np.array([2, 2])
    score = lof.decision_function(single_sample)

    assert score.shape == (1,)
    print(f"✓ Single sample prediction: score = {score[0]:.3f}")

    # Test single sample (2D array with one row)
    single_sample_2d = np.array([[2, 2]])
    score_2d = lof.decision_function(single_sample_2d)

    assert score_2d.shape == (1,)
    np.testing.assert_array_almost_equal(score, score_2d)
    print("✓ Single sample prediction consistent between 1D and 2D input")


def test_different_dimensions():
    """Test with different feature dimensions."""
    print("\n=== Testing Different Dimensions ===")

    # Test various dimensions
    for dim in [1, 3, 5, 10]:
        np.random.seed(100 + dim)
        X = np.random.normal(0, 1, (30, dim))

        # Add some outliers
        outliers = np.random.uniform(-3, 3, (3, dim))
        X = np.vstack([X, outliers])

        lof = FogMLLOF(k=5)
        lof.fit(X)
        scores = lof.decision_function(X)

        print(
            f"✓ Dimension {dim}: scores shape {scores.shape}, range [{scores.min():.3f}, {scores.max():.3f}]"
        )


if __name__ == "__main__":
    print("Testing FogML LOF Python Wrapper")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_convenience_function()
        test_edge_cases()
        test_single_sample_prediction()
        test_different_dimensions()

        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
