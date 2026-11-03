"""
Failure-mode analysis for semdiff.

Systematically tests known hard cases:
  1. Negation flips (not → present)
  2. Number changes (subtle amount changes)
  3. Long compound sentences (segmentation errors)
  4. Poor translations / OCR noise
  5. Modal verb shifts (may → must)
  6. Cross-lingual negation

Reports: per-failure-mode accuracy and examples of misclassifications.

Usage:
    python evaluation/failure_analysis.py
"""

from __future__ import annotations

import sys
import io

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


FAILURE_MODES = {
    "negation_flip": [
        (
            "Every entity shall submit a quarterly compliance report within 30 days.",
            "Every entity shall not submit a quarterly compliance report within 30 days.",
            "critical",
            "Explicit negation injection: 'shall' → 'shall not'",
        ),
        (
            "Banks may extend credit facilities without additional collateral.",
            "Banks may not extend credit facilities without additional collateral.",
            "critical",
            "Negation insertion changes permission to prohibition",
        ),
        (
            "No person shall engage in unauthorized transactions.",
            "Every person shall engage in unauthorized transactions.",
            "critical",
            "Double negation flip: 'No person shall' → 'Every person shall'",
        ),
    ],
    "number_change": [
        (
            "The minimum capital requirement shall not be less than Rs. 500 crore.",
            "The minimum capital requirement shall not be less than Rs. 501 crore.",
            "substantive",
            "Very small numerical change (500→501): models may miss subtle amounts",
        ),
        (
            "Submit within 30 days of the end of each quarter.",
            "Submit within 31 days of the end of each quarter.",
            "substantive",
            "Off-by-one day change: character diff may classify as cosmetic",
        ),
        (
            "Penalty of Rs. 5,00,000 per day of continuing default.",
            "Penalty of Rs. 50,00,000 per day of continuing default.",
            "substantive",
            "10x amount change using Indian formatting: character model may struggle",
        ),
    ],
    "long_compound_sentences": [
        (
            (
                "All payment service providers, including banks, non-banking financial companies, "
                "and registered payment aggregators, shall maintain adequate liquidity buffers "
                "equivalent to at least 15 per cent of their average daily net settlement obligations "
                "calculated on a rolling 30-day basis, as per the framework issued by the Reserve Bank "
                "of India from time to time, and shall report any breach of this threshold to the "
                "respective regulatory authority within 24 hours of its occurrence."
            ),
            (
                "All payment service providers, including banks, non-banking financial companies, "
                "and registered payment aggregators, must maintain adequate liquidity buffers "
                "equivalent to at least 20 per cent of their average daily net settlement obligations "
                "calculated on a rolling 30-day basis, as per the framework issued by the Reserve Bank "
                "of India from time to time, and shall report any breach of this threshold to the "
                "respective regulatory authority within 48 hours of its occurrence."
            ),
            "substantive",
            "Long compound sentence: two substantive changes (15%→20%, 24h→48h), may only catch one",
        ),
    ],
    "ocr_noise": [
        (
            "The minimum capital requirement shall not be less than Rs. 500 crore at all times.",
            "The m1n1mum cap1tal requ1rement sha11 not be 1ess than Rs. 500 crore at a11 t1mes.",
            "cosmetic",
            "OCR noise: l/1 substitution — model should recognise same meaning despite character errors",
        ),
        (
            "Every entity shall submit a quarterly compliance report within 30 days.",
            "Every ent1ty sha1l subm1t a quarterly comp1iance report w1thin 30 days.",
            "cosmetic",
            "Scattered OCR 1/l confusion — expected cosmetic but character diff may say critical",
        ),
    ],
    "modal_shift": [
        (
            "Banks may extend credit facilities to qualifying entities.",
            "Banks shall extend credit facilities to qualifying entities.",
            "critical",
            "may → shall: permission becomes mandatory obligation",
        ),
        (
            "The appointed authority must approve all applications within 60 days.",
            "The appointed authority may approve all applications within 60 days.",
            "critical",
            "must → may: mandatory becomes discretionary",
        ),
        (
            "Entities should maintain an internal audit committee.",
            "Entities must maintain an internal audit committee.",
            "critical",
            "should → must: advisory becomes compulsory",
        ),
    ],
    "cross_lingual_negation": [
        (
            "No person shall engage in foreign exchange transactions without prior approval.",
            "कोई भी व्यक्ति पूर्व अनुमति के बिना विदेशी मुद्रा लेनदेन में संलग्न नहीं होगा।",  # same meaning in Hindi
            "cosmetic",
            "Cross-lingual: EN prohibition → HI translation (same meaning)",
        ),
        (
            "No person shall engage in foreign exchange transactions without prior approval.",
            "प्रत्येक व्यक्ति पूर्व अनुमति के बिना विदेशी मुद्रा लेनदेन में संलग्न हो सकता है।",  # critical change in Hindi
            "critical",
            "Cross-lingual critical: EN prohibition → HI permission (opposite meaning)",
        ),
    ],
}


def analyze_failure_modes():
    from semdiff.classifier.rule_checker import RuleBasedChecker

    checker = RuleBasedChecker()

    print("=" * 70)
    print("FAILURE MODE ANALYSIS")
    print("=" * 70)

    total_pairs = 0
    total_correct = 0
    mode_results = {}

    for mode_name, cases in FAILURE_MODES.items():
        print(f"\n{'-' * 70}")
        print(f"Failure Mode: {mode_name.upper()}")
        print(f"{'-' * 70}")

        correct = 0
        errors = []
        for text_a, text_b, true_label, description in cases:
            flags = checker.compare(text_a, text_b)
            pred_label = checker.severity_hint(flags)
            is_correct = pred_label == true_label

            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
                errors.append({
                    "description": description,
                    "text_a": text_a[:80] + "..." if len(text_a) > 80 else text_a,
                    "text_b": text_b[:80] + "..." if len(text_b) > 80 else text_b,
                    "true": true_label,
                    "pred": pred_label,
                    "flags": flags,
                })

            print(f"  {status} [{true_label} → {pred_label}] {description[:70]}")
            if flags:
                print(f"      Flags: {flags}")

        acc = correct / len(cases) if cases else 0
        print(f"\n  Accuracy on {mode_name}: {correct}/{len(cases)} = {acc:.1%}")
        mode_results[mode_name] = {"accuracy": acc, "correct": correct, "total": len(cases)}
        total_pairs += len(cases)
        total_correct += correct

        if errors:
            print(f"\n  ❌ ERRORS:")
            for err in errors:
                print(f"\n    Description: {err['description']}")
                print(f"    Text A: {err['text_a']}")
                print(f"    Text B: {err['text_b']}")
                print(f"    True: {err['true']}  |  Predicted: {err['pred']}")
                print(f"    Rule flags triggered: {err['flags'] or '(none)'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Mode':<30} {'Correct':>10} {'Total':>10} {'Accuracy':>10}")
    print("-" * 62)
    for mode_name, stats in mode_results.items():
        print(f"{mode_name:<30} {stats['correct']:>10} {stats['total']:>10} {stats['accuracy']:>9.1%}")
    print("-" * 62)
    overall = total_correct / total_pairs if total_pairs > 0 else 0
    print(f"{'OVERALL':<30} {total_correct:>10} {total_pairs:>10} {overall:>9.1%}")

    print(f"""
Observations:
  • Negation flips: Rule-based checker handles "shall not" and "No person → Every person" well.
  • Number changes: Works for large/obvious changes; subtle off-by-one may be missed.
  • Long compound sentences: Multiple simultaneous changes may only be partially captured.
  • OCR noise: Character-level differences fool the rule checker (similarity is low).
  • Modal shifts: "may → shall" is well-handled; "should → must" may be missed if
    "should" is treated as neither strong nor weak.
  • Cross-lingual negation: LaBSE captures semantic similarity; rule flags on non-English
    text depend on Indic keyword matching completeness.

Mitigations:
  • Use DistilBERT classifier (combined mode) to capture long-distance semantic shifts.
  • Add OCR pre-processing (spell correction) before ingestion.
  • Expand modal/negation lexicons for Hindi and Tamil.
  • For multi-change sentences, report all detected flags even when individual
    severity is "cosmetic".
""")


if __name__ == "__main__":
    analyze_failure_modes()

