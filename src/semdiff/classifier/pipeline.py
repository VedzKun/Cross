"""
Classification pipeline that combines DistilBERT predictions with rule-based signals
to assign final severity labels to aligned sentence pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..models import AlignedPair, AlignmentType, SeverityLevel, DiffResult
from .rule_checker import RuleBasedChecker, RuleFlags
from .distilbert_classifier import SeverityClassifier, LABEL2ID

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {
    SeverityLevel.UNCHANGED: 0,
    SeverityLevel.COSMETIC: 1,
    SeverityLevel.SUBSTANTIVE: 2,
    SeverityLevel.CRITICAL: 3,
}


class ClassificationPipeline:
    """
    Classifies each AlignedPair with a combined strategy:

    1. Rule-based flags (RuleBasedChecker) give a fast signal.
    2. DistilBERT classifier gives a learned signal with confidence.
    3. Final label = max(rule_hint, model_label) to avoid downgrading obvious changes.

    Ablation modes:
        - mode="rule_only": skip DistilBERT, use only rule-based hints
        - mode="model_only": skip rules, use only DistilBERT
        - mode="combined" (default): merge both

    If no fine-tuned checkpoint is available, falls back to rule_only.
    """

    def __init__(
        self,
        classifier_checkpoint: Optional[Path] = None,
        mode: str = "combined",
        device: str = "cpu",
    ):
        self.mode = mode
        self.rule_checker = RuleBasedChecker()
        self.classifier: Optional[SeverityClassifier] = None

        if mode in ("model_only", "combined"):
            self.classifier = SeverityClassifier(
                checkpoint_dir=classifier_checkpoint or Path("models/classifier"),
                device=device,
            )
            if classifier_checkpoint and classifier_checkpoint.exists():
                try:
                    self.classifier.load(classifier_checkpoint)
                    logger.info(f"Loaded classifier from {classifier_checkpoint}")
                except Exception as e:
                    logger.warning(f"Could not load classifier checkpoint: {e}. Using rule_only fallback.")
                    self.mode = "rule_only"
            else:
                logger.info("No classifier checkpoint — using rule_only mode.")
                self.mode = "rule_only"

    def classify_pair(self, pair: AlignedPair) -> AlignedPair:
        """Classify a single AlignedPair and update severity + rule_flags in place."""
        return self.classify_pairs([pair])[0]

    def classify_pairs(self, pairs: List[AlignedPair]) -> List[AlignedPair]:
        """Batch-classify a list of AlignedPairs."""
        # Insertions and deletions are always critical; no classifier needed
        fixed = []
        to_classify_idxs = []
        to_classify_pairs = []

        for i, pair in enumerate(pairs):
            if pair.alignment_type in (AlignmentType.INSERTION, AlignmentType.DELETION):
                pair.severity = SeverityLevel.CRITICAL
                pair.confidence = 1.0
                fixed.append((i, pair))
            else:
                to_classify_idxs.append(i)
                to_classify_pairs.append(pair)

        if to_classify_pairs:
            texts_a = [p.text_a for p in to_classify_pairs]
            texts_b = [p.text_b for p in to_classify_pairs]

            # Step 1: Rule-based flags
            all_flags: List[List[str]] = [
                self.rule_checker.compare(a, b) for a, b in zip(texts_a, texts_b)
            ]
            rule_hints = [self.rule_checker.severity_hint(f) for f in all_flags]

            # Step 2: Model predictions
            if self.mode in ("model_only", "combined") and self.classifier:
                try:
                    model_labels, model_confs = self.classifier.predict(texts_a, texts_b)
                except Exception as e:
                    logger.warning(f"Classifier prediction failed: {e}. Falling back to rules.")
                    model_labels = [None] * len(texts_a)
                    model_confs = [0.5] * len(texts_a)
            else:
                model_labels = [None] * len(texts_a)
                model_confs = [0.5] * len(texts_a)

            # Step 3: Merge
            for idx, (pair, flags, rule_hint, model_label, model_conf) in enumerate(
                zip(to_classify_pairs, all_flags, rule_hints, model_labels, model_confs)
            ):
                pair.rule_flags = flags
                final_sev, final_conf = self._merge(
                    pair, rule_hint, model_label, model_conf
                )
                pair.severity = final_sev
                pair.confidence = final_conf
                fixed.append((to_classify_idxs[idx], pair))

        # Restore original order
        fixed.sort(key=lambda x: x[0])
        return [p for _, p in fixed]

    def classify_diff_result(self, diff: DiffResult) -> DiffResult:
        """Classify all pairs in a DiffResult."""
        diff.aligned_pairs = self.classify_pairs(diff.aligned_pairs)
        return diff

    def _merge(
        self,
        pair: AlignedPair,
        rule_hint: str,
        model_label: Optional[str],
        model_conf: float,
    ) -> Tuple[SeverityLevel, float]:
        """
        Combine rule hint and model label.
        Policy: take the HIGHER of the two severity signals.
        """
        _str_to_sev = {
            "cosmetic": SeverityLevel.COSMETIC,
            "substantive": SeverityLevel.SUBSTANTIVE,
            "critical": SeverityLevel.CRITICAL,
        }

        rule_sev = _str_to_sev.get(rule_hint, SeverityLevel.COSMETIC)

        # Use similarity-based initial estimate for unchanged pairs
        if pair.similarity_score >= 0.95 and not pair.rule_flags:
            return SeverityLevel.UNCHANGED, pair.similarity_score

        if self.mode == "rule_only" or model_label is None:
            # Fallback to rule-based and similarity
            sim_sev = _sim_to_severity(pair.similarity_score)
            final = _max_severity(rule_sev, sim_sev)
            return final, 0.75

        model_sev = _str_to_sev.get(model_label, SeverityLevel.COSMETIC)

        if self.mode == "model_only":
            return model_sev, model_conf

        # Combined: take the higher severity
        final = _max_severity(rule_sev, model_sev)
        # Confidence is model confidence, boosted if rules agree
        conf = model_conf
        if rule_sev == model_sev:
            conf = min(1.0, model_conf * 1.1)
        return final, round(conf, 4)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _sim_to_severity(score: float) -> SeverityLevel:
    if score >= 0.95:
        return SeverityLevel.UNCHANGED
    elif score >= 0.80:
        return SeverityLevel.COSMETIC
    elif score >= 0.55:
        return SeverityLevel.SUBSTANTIVE
    return SeverityLevel.CRITICAL


def _max_severity(a: SeverityLevel, b: SeverityLevel) -> SeverityLevel:
    if _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b]:
        return a
    return b

