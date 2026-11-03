"""
semdiff — Cross-Lingual Semantic Diff for Regulatory Documents.
"""

__version__ = "1.0.0"
__author__ = "semdiff"

from .models import (
    SentenceItem, Document, AlignedPair, DiffResult,
    SeverityLevel, AlignmentType, ClauseTimeline, ClauseLineageHop,
)
from .config import default_settings

__all__ = [
    "SentenceItem", "Document", "AlignedPair", "DiffResult",
    "SeverityLevel", "AlignmentType", "ClauseTimeline", "ClauseLineageHop",
    "default_settings",
]
