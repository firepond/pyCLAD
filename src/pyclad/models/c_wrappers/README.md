# FogML LOF Python Wrapper

This directory contains a Python wrapper for the high-performance C implementation of the Local Outlier Factor (LOF) algorithm from FogML.

## Overview

The FogML LOF implementation provides a fast, memory-efficient outlier detection algorithm suitable for embedded systems and high-performance applications. This Python wrapper makes it easy to use from Python while maintaining the performance benefits of the C implementation.

## Files

- `fogml_lof_wrapper.py` - Main Python wrapper class
- `test_lof_wrapper.py` - Comprehensive test suite
- `c_source/` - C implementation files
  - `fogml_lof.c` - LOF algorithm implementation
  - `fogml_lof.h` - Header file with struct definitions
  - `Makefile` - Build configuration
  - `lib/fogml_lof.so` - Compiled shared library

## Installation

1. Ensure you have a C compiler (gcc) installed
2. Compile the C library:
   ```bash
   cd c_source
   make
   ```
3. Install Python dependencies:
   ```bash
   pip install numpy
   ```

## Usage

### Basic Usage

```python
import numpy as np
from fogml_lof_wrapper import FogMLLOF

# Generate sample data
X = np.random.normal(0, 1, (100, 2))
X = np.vstack([X, [[3, 3], [-3, -3]]])  # Add outliers

# Create and fit LOF model
lof = FogMLLOF(k=5)
lof.fit(X)

# Get outlier scores
scores = lof.decision_function(X)
print(f"LOF scores range: [{scores.min():.3f}, {scores.max():.3f}]")

# Predict outliers (1 = outlier, 0 = normal)
predictions = lof.predict(X, threshold=1.5)
print(f"Found {predictions.sum()} outliers")
```

### Convenience Function

For quick outlier detection:

```python
from fogml_lof_wrapper import lof_outlier_detection

predictions, scores = lof_outlier_detection(X, k=5, threshold=1.5)
```

### Advanced Usage

```python
# Access internal LOF components
k_distances = lof.get_k_distances()          # k-distance for each training point
lrd_values = lof.get_local_reachability_densities()  # Local reachability density
training_scores = lof.get_training_scores()  # LOF scores for training data

# Single sample prediction
single_sample = np.array([2.5, 2.5])
score = lof.decision_function(single_sample)
```

## API Reference

### FogMLLOF Class

#### Constructor
```python
FogMLLOF(k=5)
```
- `k` (int): Number of nearest neighbors to consider (default: 5)

#### Methods

##### fit(X)
Fit the LOF model to training data.
- `X` (array-like): Training data of shape (n_samples, n_features)
- Returns: self

##### decision_function(X)
Compute LOF scores for input samples.
- `X` (array-like): Input samples of shape (n_samples, n_features)
- Returns: np.ndarray of LOF scores (higher = more anomalous)

##### predict(X, threshold=1.5)
Predict outliers in input samples.
- `X` (array-like): Input samples
- `threshold` (float): LOF threshold for outlier classification
- Returns: np.ndarray of binary predictions (1 = outlier, 0 = normal)

##### fit_predict(X, threshold=1.5)
Fit model and predict outliers in training data.
- `X` (array-like): Training data
- `threshold` (float): LOF threshold
- Returns: np.ndarray of binary predictions

##### get_k_distances()
Get k-distances for training data points.
- Returns: np.ndarray or None

##### get_local_reachability_densities()
Get local reachability densities for training data points.
- Returns: np.ndarray or None

##### get_training_scores()
Get LOF scores for training data points.
- Returns: np.ndarray or None

### Convenience Function

##### lof_outlier_detection(X, k=5, threshold=1.5)
Quick outlier detection function.
- `X` (array-like): Input data
- `k` (int): Number of neighbors
- `threshold` (float): Outlier threshold
- Returns: tuple of (predictions, scores)

## Algorithm Details

The Local Outlier Factor (LOF) algorithm:

1. **K-nearest neighbors**: For each point, find k nearest neighbors
2. **K-distance**: Distance to the k-th nearest neighbor
3. **Reachability distance**: Max of k-distance and actual distance
4. **Local reachability density (LRD)**: Inverse of average reachability distance
5. **LOF score**: Ratio of average LRD of neighbors to point's own LRD

Points with LOF scores significantly greater than 1 are considered outliers.

## Performance Notes

- The C implementation is optimized for performance and memory efficiency
- Time complexity: O(n²) for training, O(n) for prediction per sample
- Memory usage is proportional to the training set size
- Suitable for embedded systems and real-time applications

## Limitations

- Maximum k value is hardcoded to 10 in the C implementation
- Uses Euclidean distance only
- Training data is stored in memory for prediction

## Example Output

```
Dataset shape: (102, 2)
Model fitted: FogMLLOF(k=5, fitted)
LOF scores range: [1.070, 2.579]
Found 7 outliers out of 102 samples
K-distances range: [0.239, 2.541]  
LRD values range: [0.503, 5.386]
```

## Testing

Run the test suite to verify functionality:

```bash
python test_lof_wrapper.py
```

The test suite covers:
- Basic functionality and API compliance
- Edge cases and error handling
- Different data dimensions
- Single sample prediction
- Convenience functions

## License

Copyright 2021 FogML
Licensed under the Apache License, Version 2.0
