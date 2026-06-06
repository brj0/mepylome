"""Methylation analysis tools module.

Provides a class for performing methylation analysis and a Dash-based browser
application.
"""

from .classifiers import TrainedClassifier
from .core import MethylAnalysis

__all__ = ["MethylAnalysis", "TrainedClassifier"]
