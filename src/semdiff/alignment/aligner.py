"""
Needleman-Wunsch style semantic alignment for regulatory documents.

Supports 1-1, 1-many (split), many-1 (merge), insertion, and deletion alignment types.
Uses cosine-similarity matrix derived from LaBSE embeddings as the scoring function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..models import (
    SentenceItem, AlignedPair, AlignmentType, DiffResult, SeverityLevel
)
from ..config import default_settings

logger = logging.getLogger(__name__)


class TraceOp(str, Enum):
    MATCH = "M"
    GAP_A = "GA"   # gap in A → insertion in B
    GAP_B = "GB"   # gap in B → deletion from A


@dataclass
class DPCell:
    score: float = -1e9
    op: Optional[TraceOp] = None
    # For multi-row extensions (split / merge windows)
    merge_len_a: int = 1  # how many A rows consumed
    merge_len_b: int = 1  # how many B cols consumed


class SemanticAligner:
    """
    Aligns two lists of sentences using a DP variant of Needleman-Wunsch
    scored by LaBSE cosine similarity.

    Alignment types resolved:
      - 1-1  MATCH: sim ≥ sim_threshold
      - insert: gap in A (sentence added in B)
      - delete: gap in B (sentence removed from A)
      - 1-many (split): one A maps best to k consecutive B sentences
      - many-1 (merge): k consecutive A sentences map to one B sentence
    """

    def __init__(
        self,
        sim_threshold: float = default_settings.sim_threshold,
        gap_open: float = default_settings.gap_open_penalty,
        gap_extend: float = default_settings.gap_extend_penalty,
        split_merge_threshold: float = default_settings.split_merge_threshold,
        max_split_merge_window: int = 4,
    ):
        self.sim_threshold = sim_threshold
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.split_merge_threshold = split_merge_threshold
        self.max_window = max_split_merge_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        sentences_a: List[SentenceItem],
        sentences_b: List[SentenceItem],
        sim_matrix: np.ndarray,
        doc_a_id: str = "docA",
        doc_b_id: str = "docB",
    ) -> DiffResult:
        """
        Run DP alignment and return a DiffResult containing all AlignedPairs.
        """
        if sim_matrix.shape != (len(sentences_a), len(sentences_b)):
            raise ValueError(
                f"sim_matrix shape {sim_matrix.shape} does not match "
                f"({len(sentences_a)}, {len(sentences_b)})"
            )

        raw_pairs = self._dp_align(sentences_a, sentences_b, sim_matrix)
        # Post-process to detect split/merge blocks
        pairs = self._resolve_split_merge(raw_pairs, sentences_a, sentences_b, sim_matrix)

        import datetime
        result = DiffResult(
            doc_a_id=doc_a_id,
            doc_b_id=doc_b_id,
            aligned_pairs=pairs,
            stats=self._compute_stats(pairs),
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        return result

    # ------------------------------------------------------------------
    # DP core
    # ------------------------------------------------------------------

    def _dp_align(
        self,
        sents_a: List[SentenceItem],
        sents_b: List[SentenceItem],
        sim: np.ndarray,
    ) -> List[AlignedPair]:
        """Classic linear-gap Needleman-Wunsch traceback."""
        n, m = len(sents_a), len(sents_b)

        # Score matrix
        dp = np.full((n + 1, m + 1), -1e9, dtype=np.float64)
        # Traceback: 0=diag, 1=up(gap_b), 2=left(gap_a)
        tb = np.zeros((n + 1, m + 1), dtype=np.int8)

        dp[0, 0] = 0.0
        # Initialize first column (gaps in B = deletions)
        for i in range(1, n + 1):
            dp[i, 0] = -self.gap_open - (i - 1) * self.gap_extend
            tb[i, 0] = 1  # up

        # Initialize first row (gaps in A = insertions)
        for j in range(1, m + 1):
            dp[0, j] = -self.gap_open - (j - 1) * self.gap_extend
            tb[0, j] = 2  # left

        # Fill
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score_diag = dp[i - 1, j - 1] + float(sim[i - 1, j - 1])
                score_up = dp[i - 1, j] - (self.gap_open if tb[i - 1, j] != 1 else self.gap_extend)
                score_left = dp[i, j - 1] - (self.gap_open if tb[i, j - 1] != 2 else self.gap_extend)

                best = max(score_diag, score_up, score_left)
                dp[i, j] = best
                if best == score_diag:
                    tb[i, j] = 0
                elif best == score_up:
                    tb[i, j] = 1
                else:
                    tb[i, j] = 2

        # Traceback
        alignment: List[Tuple[Optional[int], Optional[int]]] = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and tb[i, j] == 0:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and tb[i, j] == 1:
                alignment.append((i - 1, None))  # deletion
                i -= 1
            else:
                alignment.append((None, j - 1))  # insertion
                j -= 1

        alignment.reverse()

        # Convert to AlignedPair objects
        pairs: List[AlignedPair] = []
        pair_idx = 0
        for a_idx, b_idx in alignment:
            if a_idx is not None and b_idx is not None:
                score = float(sim[a_idx, b_idx])
                atype = AlignmentType.ONE_TO_ONE
                severity = self._classify_similarity(score)
                pairs.append(AlignedPair(
                    pair_id=f"p{pair_idx:04d}",
                    alignment_type=atype,
                    sentences_a=[sents_a[a_idx]],
                    sentences_b=[sents_b[b_idx]],
                    similarity_score=score,
                    severity=severity,
                    confidence=score,
                ))
            elif a_idx is not None:
                pairs.append(AlignedPair(
                    pair_id=f"p{pair_idx:04d}",
                    alignment_type=AlignmentType.DELETION,
                    sentences_a=[sents_a[a_idx]],
                    sentences_b=[],
                    similarity_score=0.0,
                    severity=SeverityLevel.CRITICAL,
                    confidence=1.0,
                ))
            else:
                pairs.append(AlignedPair(
                    pair_id=f"p{pair_idx:04d}",
                    alignment_type=AlignmentType.INSERTION,
                    sentences_a=[],
                    sentences_b=[sents_b[b_idx]],
                    similarity_score=0.0,
                    severity=SeverityLevel.CRITICAL,
                    confidence=1.0,
                ))
            pair_idx += 1

        return pairs

    def _classify_similarity(self, score: float) -> SeverityLevel:
        """Map raw similarity score to initial severity label (before classifier override)."""
        if score >= 0.95:
            return SeverityLevel.UNCHANGED
        elif score >= 0.80:
            return SeverityLevel.COSMETIC
        elif score >= 0.55:
            return SeverityLevel.SUBSTANTIVE
        else:
            return SeverityLevel.CRITICAL

    # ------------------------------------------------------------------
    # Split / Merge post-processing
    # ------------------------------------------------------------------

    def _resolve_split_merge(
        self,
        pairs: List[AlignedPair],
        sents_a: List[SentenceItem],
        sents_b: List[SentenceItem],
        sim: np.ndarray,
    ) -> List[AlignedPair]:
        """
        Scan consecutive (delete + insert) blocks and test if collapsing them
        into a 1-many or many-1 relationship yields a higher average similarity.
        """
        resolved: List[AlignedPair] = []
        i = 0
        pair_idx = 0

        while i < len(pairs):
            p = pairs[i]

            # Look ahead for a run of deletions followed by insertions (or vice-versa)
            # that could form a split (1-many) or merge (many-1)
            if p.alignment_type == AlignmentType.DELETION:
                del_block = [p]
                j = i + 1
                while j < len(pairs) and pairs[j].alignment_type == AlignmentType.DELETION and len(del_block) < self.max_window:
                    del_block.append(pairs[j])
                    j += 1

                ins_block: List[AlignedPair] = []
                k = j
                while k < len(pairs) and pairs[k].alignment_type == AlignmentType.INSERTION and len(ins_block) < self.max_window:
                    ins_block.append(pairs[k])
                    k += 1

                merged = self._try_merge(del_block, ins_block, sim, pair_idx)
                if merged:
                    resolved.extend(merged)
                    pair_idx += len(merged)
                    i = k
                    continue

            if p.alignment_type == AlignmentType.INSERTION:
                ins_block = [p]
                j = i + 1
                while j < len(pairs) and pairs[j].alignment_type == AlignmentType.INSERTION and len(ins_block) < self.max_window:
                    ins_block.append(pairs[j])
                    j += 1

                del_block = []
                k = j
                while k < len(pairs) and pairs[k].alignment_type == AlignmentType.DELETION and len(del_block) < self.max_window:
                    del_block.append(pairs[k])
                    k += 1

                merged = self._try_merge(del_block, ins_block, sim, pair_idx)
                if merged:
                    resolved.extend(merged)
                    pair_idx += len(merged)
                    i = k
                    continue

            # Passthrough
            p_copy = p.model_copy(update={"pair_id": f"p{pair_idx:04d}"})
            resolved.append(p_copy)
            pair_idx += 1
            i += 1

        return resolved

    def _try_merge(
        self,
        del_block: List[AlignedPair],
        ins_block: List[AlignedPair],
        sim: np.ndarray,
        pair_idx: int,
    ) -> Optional[List[AlignedPair]]:
        """
        Attempt to collapse deletion+insertion blocks into split/merge alignments.
        Returns new pairs if the threshold is met, else None.
        """
        if not del_block or not ins_block:
            return None

        a_sents = [s for p in del_block for s in p.sentences_a]
        b_sents = [s for p in ins_block for s in p.sentences_b]

        if not a_sents or not b_sents:
            return None

        a_idxs = [s.index for s in a_sents]
        b_idxs = [s.index for s in b_sents]

        # Compute sub-matrix
        try:
            sub_sim = sim[np.ix_(a_idxs, b_idxs)]
        except IndexError:
            return None

        avg_sim = float(np.mean(sub_sim))
        if avg_sim < self.split_merge_threshold:
            return None

        # Determine type
        if len(a_sents) == 1 and len(b_sents) > 1:
            atype = AlignmentType.ONE_TO_MANY
        elif len(a_sents) > 1 and len(b_sents) == 1:
            atype = AlignmentType.MANY_TO_ONE
        elif len(a_sents) > 1 and len(b_sents) > 1:
            # Both non-trivial — treat as substantive replacement
            atype = AlignmentType.ONE_TO_ONE
        else:
            atype = AlignmentType.ONE_TO_ONE

        severity = self._classify_similarity(avg_sim)
        return [AlignedPair(
            pair_id=f"p{pair_idx:04d}",
            alignment_type=atype,
            sentences_a=a_sents,
            sentences_b=b_sents,
            similarity_score=avg_sim,
            severity=severity,
            confidence=avg_sim,
        )]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_stats(self, pairs: List[AlignedPair]) -> Dict:
        counts: Dict[str, int] = {t.value: 0 for t in AlignmentType}
        severity_counts: Dict[str, int] = {s.value: 0 for s in SeverityLevel}
        for p in pairs:
            counts[p.alignment_type.value] += 1
            severity_counts[p.severity.value] += 1
        return {
            "total_pairs": len(pairs),
            "alignment_type_counts": counts,
            "severity_counts": severity_counts,
        }


def align_documents(
    sentences_a: List[SentenceItem],
    sentences_b: List[SentenceItem],
    embedder,
    doc_a_id: str = "docA",
    doc_b_id: str = "docB",
    aligner: Optional[SemanticAligner] = None,
) -> DiffResult:
    """
    High-level convenience: embed both sentence lists and run alignment.
    """
    if aligner is None:
        aligner = SemanticAligner()

    texts_a = [s.text for s in sentences_a]
    texts_b = [s.text for s in sentences_b]

    emb_a = embedder.embed_sentences(texts_a)
    emb_b = embedder.embed_sentences(texts_b)
    sim_matrix = embedder.compute_similarity_matrix(emb_a, emb_b)

    return aligner.align(sentences_a, sentences_b, sim_matrix, doc_a_id, doc_b_id)
