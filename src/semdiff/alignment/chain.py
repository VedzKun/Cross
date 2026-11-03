"""
Multi-version amendment chain analysis.

Tracks how each clause evolves across a sequence of document versions
(v1 → v2 → v3 → ...) and produces per-clause timelines.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from ..models import (
    Document, DiffResult, AlignedPair, AlignmentType, SeverityLevel,
    ClauseLineageHop, ClauseTimeline, SentenceItem,
)
from .aligner import align_documents, SemanticAligner

logger = logging.getLogger(__name__)


class AmendmentChain:
    """
    Applies the diff pipeline to a sequence of document versions and
    tracks per-clause amendment lineage.

    Usage:
        chain = AmendmentChain(embedder=embedder)
        chain.add_version(doc_v1)
        chain.add_version(doc_v2)
        chain.add_version(doc_v3)
        timelines = chain.get_timelines()
        diffs = chain.get_diffs()         # list of DiffResult, one per consecutive pair
    """

    def __init__(
        self,
        embedder,
        aligner: Optional[SemanticAligner] = None,
    ):
        self.embedder = embedder
        self.aligner = aligner or SemanticAligner()
        self._versions: List[Document] = []
        self._diffs: List[DiffResult] = []

    def add_version(self, doc: Document) -> None:
        """Add the next document version and compute the diff vs. the previous."""
        self._versions.append(doc)
        if len(self._versions) >= 2:
            prev = self._versions[-2]
            curr = self._versions[-1]
            logger.info(f"Aligning {prev.doc_id} ({prev.version}) → {curr.doc_id} ({curr.version})")
            diff = align_documents(
                prev.sentences,
                curr.sentences,
                self.embedder,
                doc_a_id=prev.doc_id,
                doc_b_id=curr.doc_id,
                aligner=self.aligner,
            )
            self._diffs.append(diff)

    def get_diffs(self) -> List[DiffResult]:
        return self._diffs

    def get_timelines(self) -> List[ClauseTimeline]:
        """
        Build per-clause timelines by tracing each clause through consecutive diffs.

        Algorithm:
          - Seeds: clauses from version 1.
          - For each consecutive pair diff, map clause_id forward through alignment.
          - Record one ClauseLineageHop per transition.
        """
        if not self._versions or not self._diffs:
            return []

        # Build a map: clause_id → ClauseTimeline
        timelines: Dict[str, ClauseTimeline] = {}

        # Seed from the first version
        for sent in self._versions[0].sentences:
            cid = sent.clause_id or f"_sent_{sent.index}"
            if cid not in timelines:
                timelines[cid] = ClauseTimeline(clause_id=cid)

        # Walk through each consecutive diff
        for diff_idx, diff in enumerate(self._diffs):
            doc_a = self._versions[diff_idx]
            doc_b = self._versions[diff_idx + 1]

            version_from = doc_a.version
            version_to = doc_b.version

            # Map sentence index → clause_id for doc_a and doc_b
            a_sent_map: Dict[int, str] = {
                s.index: (s.clause_id or f"_sent_{s.index}")
                for s in doc_a.sentences
            }
            b_sent_map: Dict[int, str] = {
                s.index: (s.clause_id or f"_sent_{s.index}")
                for s in doc_b.sentences
            }

            # Ensure all B clauses have timelines (for newly inserted clauses)
            for sent in doc_b.sentences:
                cid = sent.clause_id or f"_sent_{sent.index}"
                if cid not in timelines:
                    timelines[cid] = ClauseTimeline(clause_id=cid)

            for pair in diff.aligned_pairs:
                hop = self._pair_to_hop(pair, version_from, version_to, a_sent_map, b_sent_map)
                if hop is None:
                    continue

                # Determine which clause_id to update
                if pair.sentences_a:
                    primary_cid = a_sent_map.get(pair.sentences_a[0].index, pair.sentences_a[0].clause_id or "")
                elif pair.sentences_b:
                    primary_cid = b_sent_map.get(pair.sentences_b[0].index, pair.sentences_b[0].clause_id or "")
                else:
                    continue

                if primary_cid not in timelines:
                    timelines[primary_cid] = ClauseTimeline(clause_id=primary_cid)
                timelines[primary_cid].history.append(hop)

            # Mark final status of timelines
            b_clause_ids = set(b_sent_map.values())
            for cid, tl in timelines.items():
                if tl.final_status == "active" and diff_idx == len(self._diffs) - 1:
                    if cid.startswith("_sent_"):
                        # Numeric synthetic id — check if it still exists
                        if cid not in b_clause_ids:
                            tl.final_status = "deleted"
                    elif cid not in b_clause_ids:
                        tl.final_status = "deleted"

        return list(timelines.values())

    def _pair_to_hop(
        self,
        pair: AlignedPair,
        version_from: str,
        version_to: str,
        a_map: Dict[int, str],
        b_map: Dict[int, str],
    ) -> Optional[ClauseLineageHop]:
        if pair.alignment_type == AlignmentType.INSERTION:
            # New clause added
            return ClauseLineageHop(
                from_version=version_from,
                to_version=version_to,
                alignment_type=pair.alignment_type,
                severity=SeverityLevel.CRITICAL,
                text_before="[INSERTED]",
                text_after=pair.text_b,
                similarity=0.0,
                confidence=1.0,
            )
        elif pair.alignment_type == AlignmentType.DELETION:
            return ClauseLineageHop(
                from_version=version_from,
                to_version=version_to,
                alignment_type=pair.alignment_type,
                severity=SeverityLevel.CRITICAL,
                text_before=pair.text_a,
                text_after="[DELETED]",
                similarity=0.0,
                confidence=1.0,
            )
        else:
            return ClauseLineageHop(
                from_version=version_from,
                to_version=version_to,
                alignment_type=pair.alignment_type,
                severity=pair.severity,
                text_before=pair.text_a,
                text_after=pair.text_b,
                similarity=pair.similarity_score,
                confidence=pair.confidence,
            )

    def summary(self) -> Dict:
        """Return high-level summary across the amendment chain."""
        timelines = self.get_timelines()
        status_counts: Dict[str, int] = defaultdict(int)
        for tl in timelines:
            status_counts[tl.final_status] += 1

        total_hops = sum(len(tl.history) for tl in timelines)
        critical_hops = sum(
            sum(1 for h in tl.history if h.severity == SeverityLevel.CRITICAL)
            for tl in timelines
        )
        return {
            "versions": [d.version for d in self._versions],
            "num_versions": len(self._versions),
            "num_diffs": len(self._diffs),
            "num_clause_timelines": len(timelines),
            "clause_final_status": dict(status_counts),
            "total_hops": total_hops,
            "critical_hops": critical_hops,
        }

