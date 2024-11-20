"""Methylation analysis tools module.

Provides a class for performing methylation analysis and a Dash-based browser
application.
"""

from .methyl import MethylAnalysis
from .methyl_clf import TrainedClassifier

__all__ = ["MethylAnalysis", "TrainedClassifier"]
