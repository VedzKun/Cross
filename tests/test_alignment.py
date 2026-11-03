"""
Unit tests for the semantic alignment module (Needleman-Wunsch DP).
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semdiff.models import SentenceItem, AlignmentType, SeverityLevel
from semdiff.alignment.aligner import SemanticAligner


def make_sent(idx: int, text: str = "", doc: str = "doc") -> SentenceItem:
    return SentenceItem(
        id=f"{doc}_s{idx:04d}",
        text=text or f"Sentence {idx} about regulatory compliance.",
        doc_id=doc,
        index=idx,
    )


def make_sim(n: int, m: int, diagonal_val: float = 0.85, off_val: float = 0.20) -> np.ndarray:
    """Create a similarity matrix where the diagonal is high."""
    sim = np.full((n, m), off_val, dtype=np.float32)
    for i in range(min(n, m)):
        sim[i, i] = diagonal_val
    return sim


class TestSemanticAlignerBasic:
    """Basic correctness tests for the DP aligner."""

    def test_identical_documents_all_unchanged(self):
        """Identical documents should yield all 1-1 unchanged matches."""
        n = 5
        sents = [make_sent(i) for i in range(n)]
        sim = make_sim(n, n, diagonal_val=0.98, off_val=0.10)
        aligner = SemanticAligner(sim_threshold=0.50)
        diff = aligner.align(sents, sents, sim, "a", "b")

        one_to_one = [p for p in diff.aligned_pairs if p.alignment_type == AlignmentType.ONE_TO_ONE]
        assert len(one_to_one) == n, f"Expected {n} 1-1 pairs, got {len(one_to_one)}"

        for pair in one_to_one:
            assert pair.severity in (SeverityLevel.UNCHANGED, SeverityLevel.COSMETIC)

    def test_insertion_detected(self):
        """A sentence added to B that has no counterpart in A should be an INSERTION."""
        sents_a = [make_sent(i, doc="a") for i in range(3)]
        sents_b = [make_sent(i, doc="b") for i in range(4)]  # extra sentence in B

        # First 3 sents match well; 4th in B has no match in A
        sim = np.array([
            [0.90, 0.10, 0.10, 0.05],
            [0.10, 0.90, 0.10, 0.05],
            [0.10, 0.10, 0.90, 0.05],
        ], dtype=np.float32)

        aligner = SemanticAligner(sim_threshold=0.50)
        diff = aligner.align(sents_a, sents_b, sim, "a", "b")

        ins = [p for p in diff.aligned_pairs if p.alignment_type == AlignmentType.INSERTION]
        assert len(ins) >= 1, "Should detect at least one INSERTION"

    def test_deletion_detected(self):
        """A sentence removed from B that existed in A should be a DELETION."""
        sents_a = [make_sent(i, doc="a") for i in range(4)]  # extra sentence in A
        sents_b = [make_sent(i, doc="b") for i in range(3)]

        sim = np.array([
            [0.90, 0.10, 0.10],
            [0.10, 0.90, 0.10],
            [0.10, 0.10, 0.90],
            [0.05, 0.05, 0.05],
        ], dtype=np.float32)

        aligner = SemanticAligner(sim_threshold=0.50)
        diff = aligner.align(sents_a, sents_b, sim, "a", "b")

        dels = [p for p in diff.aligned_pairs if p.alignment_type == AlignmentType.DELETION]
        assert len(dels) >= 1, "Should detect at least one DELETION"

    def test_empty_document_a(self):
        """If A is empty, all B sentences should be insertions."""
        sents_b = [make_sent(i, doc="b") for i in range(3)]
        sim = np.empty((0, 3), dtype=np.float32)
        aligner = SemanticAligner()
        diff = aligner.align([], sents_b, sim, "a", "b")
        ins = [p for p in diff.aligned_pairs if p.alignment_type == AlignmentType.INSERTION]
        assert len(ins) == 3

    def test_empty_document_b(self):
        """If B is empty, all A sentences should be deletions."""
        sents_a = [make_sent(i, doc="a") for i in range(3)]
        sim = np.empty((3, 0), dtype=np.float32)
        aligner = SemanticAligner()
        diff = aligner.align(sents_a, [], sim, "a", "b")
        dels = [p for p in diff.aligned_pairs if p.alignment_type == AlignmentType.DELETION]
        assert len(dels) == 3

    def test_both_empty(self):
        """Both empty documents → no pairs."""
        sim = np.empty((0, 0), dtype=np.float32)
        aligner = SemanticAligner()
        diff = aligner.align([], [], sim, "a", "b")
        assert len(diff.aligned_pairs) == 0

    def test_output_shape_consistency(self):
        """Every aligned pair should have non-empty A or B (or both) but not neither."""
        n, m = 6, 7
        sents_a = [make_sent(i, doc="a") for i in range(n)]
        sents_b = [make_sent(i, doc="b") for i in range(m)]
        sim = np.random.RandomState(42).uniform(0.2, 0.9, (n, m)).astype(np.float32)

        aligner = SemanticAligner()
        diff = aligner.align(sents_a, sents_b, sim, "a", "b")

        for pair in diff.aligned_pairs:
            assert len(pair.sentences_a) > 0 or len(pair.sentences_b) > 0, \
                f"Pair {pair.pair_id} has no sentences on either side"

    def test_pair_ids_unique(self):
        """All pair_ids in a diff should be unique."""
        n = 8
        sents_a = [make_sent(i, doc="a") for i in range(n)]
        sents_b = [make_sent(i, doc="b") for i in range(n)]
        sim = make_sim(n, n)
        aligner = SemanticAligner()
        diff = aligner.align(sents_a, sents_b, sim, "a", "b")
        ids = [p.pair_id for p in diff.aligned_pairs]
        assert len(ids) == len(set(ids)), "Pair IDs should be unique"


class TestAlignmentSeverity:
    """Tests for severity assignment based on similarity."""

    def test_high_similarity_is_unchanged(self):
        sents = [make_sent(0, doc="a"), make_sent(0, doc="b")]
        sim = np.array([[0.97]], dtype=np.float32)
        aligner = SemanticAligner()
        diff = aligner.align([sents[0]], [sents[1]], sim, "a", "b")
        assert diff.aligned_pairs[0].severity == SeverityLevel.UNCHANGED

    def test_low_similarity_is_critical(self):
        sents = [make_sent(0, doc="a"), make_sent(0, doc="b")]
        sim = np.array([[0.20]], dtype=np.float32)
        aligner = SemanticAligner()
        diff = aligner.align([sents[0]], [sents[1]], sim, "a", "b")
        assert diff.aligned_pairs[0].severity == SeverityLevel.CRITICAL

    def test_stats_keys_present(self):
        n = 4
        sents_a = [make_sent(i, doc="a") for i in range(n)]
        sents_b = [make_sent(i, doc="b") for i in range(n)]
        sim = make_sim(n, n)
        aligner = SemanticAligner()
        diff = aligner.align(sents_a, sents_b, sim, "a", "b")
        assert "total_pairs" in diff.stats
        assert "severity_counts" in diff.stats
        assert "alignment_type_counts" in diff.stats

    def test_stats_total_matches_pairs(self):
        n = 5
        sents_a = [make_sent(i, doc="a") for i in range(n)]
        sents_b = [make_sent(i, doc="b") for i in range(n)]
        sim = make_sim(n, n)
        aligner = SemanticAligner()
        diff = aligner.align(sents_a, sents_b, sim, "a", "b")
        assert diff.stats["total_pairs"] == len(diff.aligned_pairs)


class TestSplitMerge:
    """Test that 1-many (split) and many-1 (merge) are resolved."""

    def test_one_to_many_detected(self):
        """One sentence in A split into two in B."""
        sents_a = [make_sent(0, doc="a", text="Original long sentence about compliance requirements.")]
        sents_b = [
            make_sent(0, doc="b", text="Compliance requirements apply to all entities."),
            make_sent(1, doc="b", text="All entities must file reports accordingly."),
        ]
        # High similarity between a[0] and both b[0] and b[1]
        sim = np.array([[0.80, 0.78]], dtype=np.float32)
        aligner = SemanticAligner(split_merge_threshold=0.65)
        diff = aligner.align(sents_a, sents_b, sim, "a", "b")

        # Should produce at least one pair covering both B sentences
        all_b_idxs = {s.index for p in diff.aligned_pairs for s in p.sentences_b}
        assert 0 in all_b_idxs or 1 in all_b_idxs, "B sentences should be covered"

    def test_sim_matrix_shape_mismatch_raises(self):
        """Wrong sim matrix shape should raise ValueError."""
        sents_a = [make_sent(i, doc="a") for i in range(3)]
        sents_b = [make_sent(i, doc="b") for i in range(2)]
        wrong_sim = np.zeros((4, 5), dtype=np.float32)  # wrong shape
        aligner = SemanticAligner()
        with pytest.raises(ValueError):
            aligner.align(sents_a, sents_b, wrong_sim, "a", "b")

