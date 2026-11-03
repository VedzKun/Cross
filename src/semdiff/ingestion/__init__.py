"""
Ingestion module for semdiff.
"""

from .readers import DocumentReader, read_document
from .splitter import LegalSentenceSplitter, split_legal_text

__all__ = ["DocumentReader", "read_document", "LegalSentenceSplitter", "split_legal_text"]
