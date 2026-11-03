"""
Classifier module: rule-based feature checking + DistilBERT fine-tuning.
"""

from .rule_checker import RuleBasedChecker, RuleFlags
from .pipeline import ClassificationPipeline

__all__ = ["RuleBasedChecker", "RuleFlags", "ClassificationPipeline"]

