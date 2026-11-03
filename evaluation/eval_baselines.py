"""
Baseline comparison: character-level diff (difflib), TF-IDF cosine, LaBSE-only thresholding.

Usage:
    python evaluation/eval_baselines.py --demo
    python evaluation/eval_baselines.py --test data/processed/test.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_classifier import compute_metrics, print_results, LABELS


# -------------------------------------------------------------------------
# Baseline 1: character-level diff (difflib SequenceMatcher ratio)
# -------------------------------------------------------------------------

def difflib_severity(text_a: str, text_b: str) -> str:
    ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
    if ratio >= 0.95:
        return "cosmetic"
    elif ratio >= 0.70:
        return "cosmetic"
    elif ratio >= 0.45:
        return "substantive"
    return "critical"


def baseline_difflib(texts_a: List[str], texts_b: List[str]) -> List[str]:
    return [difflib_severity(a, b) for a, b in zip(texts_a, texts_b)]


# -------------------------------------------------------------------------
# Baseline 2: TF-IDF cosine similarity
# -------------------------------------------------------------------------

def baseline_tfidf(texts_a: List[str], texts_b: List[str]) -> List[str]:
    all_texts = texts_a + texts_b
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=50000,
        sublinear_tf=True,
    )
    tfidf = vec.fit_transform(all_texts)
    n = len(texts_a)
    sims_a = tfidf[:n]
    sims_b = tfidf[n:]
    preds = []
    for i in range(n):
        sim = cosine_similarity(sims_a[i], sims_b[i])[0, 0]
        if sim >= 0.90:
            preds.append("cosmetic")
        elif sim >= 0.60:
            preds.append("substantive")
        else:
            preds.append("critical")
    return preds


# -------------------------------------------------------------------------
# Baseline 3: LaBSE thresholding (no learned classifier)
# -------------------------------------------------------------------------

def baseline_labse(
    texts_a: List[str],
    texts_b: List[str],
    thresholds: Tuple[float, float] = (0.90, 0.65),
) -> List[str]:
    """
    Classify using only LaBSE cosine similarity with fixed thresholds.
    thresholds: (cosmetic_threshold, substantive_threshold)
    Falls back to mock similarity if LaBSE is unavailable.
    """
    try:
        import socket
        socket.setdefaulttimeout(3)
        from semdiff.embedding.labse import LaBSEEmbedder
        embedder = LaBSEEmbedder(enable_cache=False)
        emb_a = embedder.embed_sentences(texts_a)
        emb_b = embedder.embed_sentences(texts_b)
        sims = np.sum(emb_a * emb_b, axis=1)
    except Exception as e:
        print(f"  [LaBSE unavailable ({type(e).__name__}): using mock similarity for demo]")
        np.random.seed(99)
        sims = np.random.uniform(0.5, 0.95, len(texts_a))

    preds = []
    cos_thresh, sub_thresh = thresholds
    for sim in sims:
        if sim >= cos_thresh:
            preds.append("cosmetic")
        elif sim >= sub_thresh:
            preds.append("substantive")
        else:
            preds.append("critical")
    return preds


# -------------------------------------------------------------------------
# Main comparison
# -------------------------------------------------------------------------

def run_demo():
    pairs = [
        # (text_a, text_b, true_label)
        ("Every entity shall submit a quarterly compliance report within 30 days.",
         "Every entity will submit a quarterly compliance report within 30 days.", "cosmetic"),
        ("The minimum capital requirement shall not be less than Rs. 500 crore.",
         "The minimum capital requirement shall not be less than Rs. 750 crore.", "substantive"),
        ("No person shall engage in foreign exchange transactions without prior approval.",
         "Every person shall engage in foreign exchange transactions without prior approval.", "critical"),
        ("Banks may extend credit facilities up to Rs. 10 crore.",
         "Banks must extend credit facilities up to Rs. 15 crore.", "critical"),
        ("The board shall meet at least four times every financial year.",
         "The board is required to meet at least four times every financial year.", "cosmetic"),
        ("The interest rate shall be reset every six months.",
         "The interest rate shall be reset every twelve months.", "substantive"),
        ("Insurance companies are required to settle claims within 30 days.",
         "Insurance companies are not required to settle claims within 30 days.", "critical"),
        ("Mutual funds shall disclose the total expense ratio on a daily basis.",
         "Mutual funds shall disclose the total expense ratio on a monthly basis.", "substantive"),
        ("The appointed auditor shall submit the report within 60 days of fiscal year end.",
         "The appointed auditor shall submit the report within 60 days of fiscal year end.", "cosmetic"),
        ("Banks shall not grant loans exceeding Rs. 25 lakh to directors.",
         "Banks shall not grant loans exceeding Rs. 50 lakh to directors.", "substantive"),
    ]

    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]
    y_true = [p[2] for p in pairs]

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    for name, preds in [
        ("difflib (char-level)", baseline_difflib(texts_a, texts_b)),
        ("TF-IDF cosine", baseline_tfidf(texts_a, texts_b)),
        ("LaBSE-only threshold", baseline_labse(texts_a, texts_b)),
    ]:
        results = compute_metrics(y_true, preds)
        print(f"\n{'─'*40}")
        print(f"Baseline: {name}")
        print(f"{'─'*40}")
        macro = results["macro"]
        print(f"  Macro F1:        {macro['f1']:.4f}")
        print(f"  Macro Precision: {macro['precision']:.4f}")
        print(f"  Macro Recall:    {macro['recall']:.4f}")
        print(f"  Accuracy:        {macro['accuracy']:.4f}")
        per_class_f1 = " | ".join(f"{l}: {results['per_class'][l]['f1']:.3f}" for l in LABELS)
        print(f"  Per-class F1: {per_class_f1}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons.")
    parser.add_argument("--test", type=Path, help="Test CSV with text_a, text_b, label")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo or not args.test:
        run_demo()
        return

    texts_a, texts_b, y_true = [], [], []
    with open(args.test, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lbl = row.get(args.label_col, "").strip()
            if lbl in LABELS:
                texts_a.append(row.get("text_a", ""))
                texts_b.append(row.get("text_b", ""))
                y_true.append(lbl)

    print(f"Loaded {len(y_true)} test pairs from {args.test}")
    run_demo.__doc__  # reuse same logic with loaded data

    for name, fn in [
        ("difflib (char-level)", lambda: baseline_difflib(texts_a, texts_b)),
        ("TF-IDF cosine", lambda: baseline_tfidf(texts_a, texts_b)),
        ("LaBSE-only threshold", lambda: baseline_labse(texts_a, texts_b)),
    ]:
        preds = fn()
        results = compute_metrics(y_true, preds)
        print(f"\n{'─'*40}")
        print(f"Baseline: {name}")
        macro = results["macro"]
        print(f"  Macro F1: {macro['f1']:.4f}  Accuracy: {macro['accuracy']:.4f}")
        print_results(results)


if __name__ == "__main__":
    main()

