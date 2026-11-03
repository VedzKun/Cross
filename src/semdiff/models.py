"""
Core data models for regulatory document representation and diff results.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    UNCHANGED = "unchanged"
    COSMETIC = "cosmetic"
    SUBSTANTIVE = "substantive"
    CRITICAL = "critical"


class AlignmentType(str, Enum):
    ONE_TO_ONE = "1-1"
    ONE_TO_MANY = "1-many"  # Split in B
    MANY_TO_ONE = "many-1"  # Merge in B
    INSERTION = "insert"    # Added in B
    DELETION = "delete"     # Removed in B


class SentenceItem(BaseModel):
    """Represents a single parsed sentence with regulatory structural metadata."""
    id: str = Field(description="Unique sentence identifier")
    text: str = Field(description="Normalized sentence text")
    doc_id: str = Field(default="", description="Document ID this belongs to")
    index: int = Field(description="0-based sentence index in document")
    clause_id: Optional[str] = Field(default=None, description="Extracted clause identifier e.g. '3(1)(a)'")
    section_title: Optional[str] = Field(default=None, description="Parent section/heading title")
    page_num: Optional[int] = Field(default=None, description="Page number in original document if available")
    char_start: Optional[int] = Field(default=None, description="Start character offset")
    char_end: Optional[int] = Field(default=None, description="End character offset")
    language: Optional[str] = Field(default=None, description="Language code e.g. 'en', 'hi', 'ta'")


class Document(BaseModel):
    """Represents an ingested legal/regulatory document."""
    doc_id: str
    version: str = "v1"
    filename: str
    file_type: str  # pdf, docx, txt
    raw_text: str
    sentences: List[SentenceItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def sentence_texts(self) -> List[str]:
        return [s.text for s in self.sentences]


class AlignedPair(BaseModel):
    """An aligned unit between document A and document B."""
    pair_id: str
    alignment_type: AlignmentType
    sentences_a: List[SentenceItem] = Field(default_factory=list)
    sentences_b: List[SentenceItem] = Field(default_factory=list)
    similarity_score: float = 0.0
    severity: SeverityLevel = SeverityLevel.UNCHANGED
    confidence: float = 1.0
    rule_flags: List[str] = Field(default_factory=list, description="Triggered rule features e.g. ['number_changed', 'negation_flipped']")
    explanation: Optional[str] = None

    @property
    def text_a(self) -> str:
        return " ".join(s.text for s in self.sentences_a)

    @property
    def text_b(self) -> str:
        return " ".join(s.text for s in self.sentences_b)


class DiffResult(BaseModel):
    """Complete diff result between two document versions."""
    doc_a_id: str
    doc_b_id: str
    aligned_pairs: List[AlignedPair] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class ClauseLineageHop(BaseModel):
    """A single hop in an amendment chain for a clause."""
    from_version: str
    to_version: str
    alignment_type: AlignmentType
    severity: SeverityLevel
    text_before: str
    text_after: str
    similarity: float
    confidence: float


class ClauseTimeline(BaseModel):
    """Lineage timeline of a specific clause across multi-version amendments."""
    clause_id: str
    history: List[ClauseLineageHop] = Field(default_factory=list)
    final_status: str = "active"  # active, deleted, merged, split
