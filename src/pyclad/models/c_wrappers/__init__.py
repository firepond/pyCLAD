"""
C Wrappers for high-performance implementations.

This package contains Python wrappers for optimized C implementations
of machine learning algorithms.
"""

from .fogml_lof_wrapper import FogMLLOF, lof_outlier_detection

__all__ = ["FogMLLOF", "lof_outlier_detection"]
__version__ = "1.0.0"
