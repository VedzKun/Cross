"""
Alignment evaluation: precision, recall, F1 against gold alignments.

Gold alignment format (JSON or CSV):
  Each row: { "pair_id": "...", "a_idx": 0, "b_idx": 2, "type": "1-1" }

Usage:
    python evaluation/eval_alignment.py --gold data/gold_alignments.json --pred reports/diff.json
    python evaluation/eval_alignment.py --demo          # runs on synthetic example
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_gold_alignments(path: Path) -> List[Dict]:
    """Load gold alignments from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_predicted_pairs(diff_json: Path) -> List[Dict]:
    """Extract aligned pairs from a semdiff JSON export."""
    with open(diff_json, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("diff", {}).get("aligned_pairs", [])


def pairs_to_set(pairs: List[Dict], a_key: str = "a_idx", b_key: str = "b_idx") -> Set[Tuple]:
    """Convert pair list to set of (a_idx, b_idx) tuples."""
    result = set()
    for p in pairs:
        a_idxs = p.get("sentences_a", [])
        b_idxs = p.get("sentences_b", [])
        if isinstance(a_idxs, list):
            a_idxs = [s.get("index", -1) for s in a_idxs if isinstance(s, dict)]
        if isinstance(b_idxs, list):
            b_idxs = [s.get("index", -1) for s in b_idxs if isinstance(s, dict)]
        for ai in (a_idxs or [-1]):
            for bi in (b_idxs or [-1]):
                result.add((ai, bi))
    return result


def compute_alignment_metrics(
    gold_pairs: Set[Tuple],
    pred_pairs: Set[Tuple],
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for alignment pairs."""
    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_demo():
    """Run a self-contained alignment evaluation demo without external files."""
    from semdiff.ingestion.splitter import split_legal_text
    from semdiff.embedding.labse import LaBSEEmbedder
    from semdiff.alignment.aligner import SemanticAligner
    import numpy as np

    print("=== Alignment Evaluation Demo ===\n")

    # Synthetic document pair with known gold alignments
    text_a = """Section 1.1: Every registered entity shall submit a quarterly compliance report within 30 days.
Section 1.2: The minimum capital requirement shall not be less than Rs. 500 crore.
Section 1.3: No person shall engage in unauthorized foreign exchange transactions.
Section 1.4: Any breach shall attract a penalty of up to Rs. 5 lakh per day."""

    text_b = """Section 1.1: Every registered entity must submit a quarterly compliance report within 30 days.
Section 1.2: The minimum capital requirement shall not be less than Rs. 750 crore.
Section 1.3: No individual shall participate in unauthorized foreign exchange activities.
Section 1.5: Additional reporting requirements apply to systemically important entities."""

    # Gold: sentences 0→0, 1→1, 2→2 match; sentence 3 deleted; sentence 3 in B is new insert
    gold_pairs = {(0, 0), (1, 1), (2, 2)}

    sents_a = split_legal_text(text_a, "doc_a")
    sents_b = split_legal_text(text_b, "doc_b")

    print(f"Sentences A: {len(sents_a)}, Sentences B: {len(sents_b)}")

    # Use mock embeddings for speed (random unit vectors)
    np.random.seed(42)
    n, m = len(sents_a), len(sents_b)
    emb_a = np.random.randn(n, 768).astype(np.float32)
    emb_b = np.random.randn(m, 768).astype(np.float32)
    # Normalize
    emb_a /= np.linalg.norm(emb_a, axis=1, keepdims=True)
    emb_b /= np.linalg.norm(emb_b, axis=1, keepdims=True)

    # Make first 3 pairs similar (simulate correct alignment)
    for i in range(min(3, n, m)):
        emb_b[i] = emb_a[i] + np.random.randn(768).astype(np.float32) * 0.05
        emb_b[i] /= np.linalg.norm(emb_b[i])

    sim = np.dot(emb_a, emb_b.T)
    aligner = SemanticAligner()
    diff = aligner.align(sents_a, sents_b, sim, "doc_a", "doc_b")

    # Extract predicted pairs
    pred_pairs: Set[Tuple] = set()
    for pair in diff.aligned_pairs:
        for sa in pair.sentences_a:
            for sb in pair.sentences_b:
                pred_pairs.add((sa.index, sb.index))

    metrics = compute_alignment_metrics(gold_pairs, pred_pairs)
    print("\nGold pairs:", gold_pairs)
    print("Pred pairs:", pred_pairs)
    print(f"\nAlignment Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate alignment quality.")
    parser.add_argument("--gold", type=Path, help="Gold alignment JSON file")
    parser.add_argument("--pred", type=Path, help="Predicted diff JSON file")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo or (not args.gold and not args.pred):
        run_demo()
        return

    gold = load_gold_alignments(args.gold)
    gold_set = {(r["a_idx"], r["b_idx"]) for r in gold}

    pred = extract_predicted_pairs(args.pred)
    pred_set = pairs_to_set(pred)

    metrics = compute_alignment_metrics(gold_set, pred_set)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

