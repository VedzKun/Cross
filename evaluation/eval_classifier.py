"""
Classifier evaluation: per-class precision, recall, F1 + confusion matrix.

Usage:
    python evaluation/eval_classifier.py --test data/processed/test.csv --checkpoint models/classifier
    python evaluation/eval_classifier.py --demo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LABELS = ["cosmetic", "substantive", "critical"]


def load_test_csv(path: Path, label_col: str = "label") -> tuple:
    texts_a, texts_b, labels = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lbl = row.get(label_col, "").strip()
            if lbl in LABELS:
                texts_a.append(row.get("text_a", ""))
                texts_b.append(row.get("text_b", ""))
                labels.append(lbl)
    return texts_a, texts_b, labels


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Per-class and macro precision/recall/F1 + confusion matrix."""
    results = {"per_class": {}, "macro": {}, "confusion_matrix": None}

    # Confusion matrix
    n = len(LABELS)
    cm = np.zeros((n, n), dtype=int)
    label2id = {l: i for i, l in enumerate(LABELS)}
    for true, pred in zip(y_true, y_pred):
        if true in label2id and pred in label2id:
            cm[label2id[true], label2id[pred]] += 1
    results["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    precision_list, recall_list, f1_list = [], [], []
    for i, lbl in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results["per_class"][lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": int(cm[i, :].sum()),
        }
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)

    results["macro"] = {
        "precision": round(np.mean(precision_list), 4),
        "recall": round(np.mean(recall_list), 4),
        "f1": round(np.mean(f1_list), 4),
        "accuracy": round(np.trace(cm) / cm.sum() if cm.sum() > 0 else 0, 4),
    }

    return results


def print_results(results: Dict):
    print("\n=== Classification Report ===\n")
    print(f"{'Label':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for lbl in LABELS:
        m = results["per_class"][lbl]
        print(f"{lbl:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
    print("-" * 55)
    macro = results["macro"]
    print(f"{'Macro avg':<15} {macro['precision']:>10.4f} {macro['recall']:>10.4f} {macro['f1']:>10.4f}")
    print(f"\nAccuracy: {macro['accuracy']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(f"{'':15}", end="")
    for lbl in LABELS:
        print(f"{lbl[:8]:>12}", end="")
    print()
    cm = results["confusion_matrix"]
    for i, row_lbl in enumerate(LABELS):
        print(f"{row_lbl:<15}", end="")
        for j in range(len(LABELS)):
            print(f"{cm[i][j]:>12}", end="")
        print()


def run_demo():
    """Demo using rule-based classifier on synthetic pairs."""
    from semdiff.classifier.rule_checker import RuleBasedChecker

    checker = RuleBasedChecker()

    pairs = [
        # (text_a, text_b, true_label)
        (
            "Every entity shall submit a quarterly compliance report within 30 days.",
            "Every entity will submit a quarterly compliance report within 30 days.",
            "cosmetic",
        ),
        (
            "The minimum capital requirement shall not be less than Rs. 500 crore.",
            "The minimum capital requirement shall not be less than Rs. 750 crore.",
            "substantive",
        ),
        (
            "No person shall engage in foreign exchange transactions without prior approval.",
            "Every person shall engage in foreign exchange transactions without prior approval.",
            "critical",
        ),
        (
            "Banks may extend credit facilities up to Rs. 10 crore.",
            "Banks must extend credit facilities up to Rs. 15 crore.",
            "critical",
        ),
        (
            "The board shall meet at least four times in every financial year.",
            "The board is required to meet at least four times in every financial year.",
            "cosmetic",
        ),
        (
            "The interest rate shall be reset every six months.",
            "The interest rate shall be reset every twelve months.",
            "substantive",
        ),
        (
            "Insurance companies are required to settle claims within 30 days.",
            "Insurance companies are not required to settle claims within 30 days.",
            "critical",
        ),
        (
            "Mutual funds shall disclose the total expense ratio on a daily basis.",
            "Mutual funds shall disclose the total expense ratio on a monthly basis.",
            "substantive",
        ),
    ]

    y_true, y_pred = [], []
    for text_a, text_b, true_lbl in pairs:
        flags = checker.compare(text_a, text_b)
        pred_lbl = checker.severity_hint(flags)
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)
        print(f"  True: {true_lbl:<12}  Pred: {pred_lbl:<12}  Flags: {flags}")

    results = compute_metrics(y_true, y_pred)
    print_results(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier on labeled test set.")
    parser.add_argument("--test", type=Path, help="Test CSV file")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/classifier"))
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--mode", default="rule_only", choices=["rule_only", "combined", "model_only"])
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    parser.add_argument("--output", type=Path, help="Save metrics JSON here")
    args = parser.parse_args()

    if args.demo or not args.test:
        run_demo()
        return

    print(f"Loading test data: {args.test}")
    texts_a, texts_b, y_true = load_test_csv(args.test, args.label_col)
    print(f"  {len(y_true)} test pairs")

    from semdiff.classifier.pipeline import ClassificationPipeline
    from semdiff.models import AlignedPair, AlignmentType, SeverityLevel, SentenceItem

    clf = ClassificationPipeline(
        mode=args.mode,
        classifier_checkpoint=args.checkpoint if args.checkpoint.exists() else None,
    )

    # Build dummy pairs for classification
    dummy_pairs = []
    for i, (ta, tb) in enumerate(zip(texts_a, texts_b)):
        pair = AlignedPair(
            pair_id=f"p{i:04d}",
            alignment_type=AlignmentType.ONE_TO_ONE,
            sentences_a=[SentenceItem(id=f"a{i}", text=ta, index=i)],
            sentences_b=[SentenceItem(id=f"b{i}", text=tb, index=i)],
            similarity_score=0.5,
            severity=SeverityLevel.COSMETIC,
        )
        dummy_pairs.append(pair)

    classified = clf.classify_pairs(dummy_pairs)
    y_pred = [p.severity.value for p in classified]

    results = compute_metrics(y_true, y_pred)
    print_results(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()

