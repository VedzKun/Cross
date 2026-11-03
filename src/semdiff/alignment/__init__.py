"""
Alignment module: Needleman-Wunsch semantic alignment and multi-version chain tracking.
"""

from .aligner import SemanticAligner, align_documents
from .chain import AmendmentChain

__all__ = ["SemanticAligner", "align_documents", "AmendmentChain"]

