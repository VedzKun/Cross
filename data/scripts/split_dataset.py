"""
Document-level train/val/test split for semdiff dataset.

Splits by document ID (not by sentence pair) to prevent data leakage.
Reads data/synthetic/pairs.csv (or a hand-labeled CSV) and writes:
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv

Usage:
    python data/scripts/split_dataset.py --input data/synthetic/pairs.csv
    python data/scripts/split_dataset.py --input data/labeled/labeled_pairs.csv --label-col label_human
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple


def load_csv(path: Path, label_col: str = "label") -> Tuple[List[Dict], str]:
    """Load CSV and return (rows, actual_label_col_used)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Resolve label column
        actual_col = label_col
        if actual_col not in fieldnames:
            # Try alternatives
            for alt in ["label", "label_human", "label_synthetic"]:
                if alt in fieldnames:
                    actual_col = alt
                    break
        for row in reader:
            lbl = row.get(actual_col, "").strip()
            if lbl in ("cosmetic", "substantive", "critical"):
                rows.append(row | {"_label": lbl})
    return rows, actual_col


def split_by_document(
    rows: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Group pairs by doc_id, then assign entire doc groups to train/val/test.
    This prevents sentence-level leakage between splits.
    """
    random.seed(seed)

    # Group by doc_id
    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_doc[row.get("doc_id", "unknown")].append(row)

    doc_ids = list(by_doc.keys())
    random.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_docs = doc_ids[:n_train]
    val_docs = doc_ids[n_train:n_train + n_val]
    test_docs = doc_ids[n_train + n_val:]

    train_rows = [r for doc in train_docs for r in by_doc[doc]]
    val_rows = [r for doc in val_docs for r in by_doc[doc]]
    test_rows = [r for doc in test_docs for r in by_doc[doc]]

    return train_rows, val_rows, test_rows


def write_split(rows: List[Dict], output_path: Path, label_col: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"WARNING: {output_path} is empty!")
        return
    # Remove internal helpers
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    fieldnames = list(clean_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    counts = Counter(r.get(label_col, r.get("_label", "")) for r in rows)
    print(f"  {output_path.name}: {len(rows)} pairs | {dict(counts)}")


def main():
    parser = argparse.ArgumentParser(description="Document-level train/val/test split.")
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/pairs.csv"))
    parser.add_argument("--label-col", default="label", help="Column name for the label")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    rows, label_col = load_csv(args.input, args.label_col)
    print(f"Loaded {len(rows)} labeled rows (label col: '{label_col}')")

    # Check label distribution
    counts = Counter(r["_label"] for r in rows)
    print(f"Label distribution: {dict(counts)}")

    train, val, test = split_by_document(rows, args.train, args.val, args.seed)

    print("\nSplit results:")
    write_split(train, args.output_dir / "train.csv", label_col)
    write_split(val, args.output_dir / "val.csv", label_col)
    write_split(test, args.output_dir / "test.csv", label_col)

    # Verify no doc leakage
    train_docs = set(r.get("doc_id") for r in train)
    val_docs = set(r.get("doc_id") for r in val)
    test_docs = set(r.get("doc_id") for r in test)
    leakage = (train_docs & val_docs) | (train_docs & test_docs) | (val_docs & test_docs)
    if leakage:
        print(f"\nWARNING: Document leakage detected! Shared doc_ids: {leakage}")
    else:
        print(f"\nNo document leakage detected. Split is clean.")


if __name__ == "__main__":
    main()

