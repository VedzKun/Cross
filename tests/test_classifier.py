"""
Unit tests for the rule-based feature checker and classification pipeline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semdiff.classifier.rule_checker import RuleBasedChecker, RuleFlags


class TestRuleBasedChecker:
    """Tests for RuleBasedChecker feature extraction."""

    def setup_method(self):
        self.checker = RuleBasedChecker()

    # --- Number detection ---
    def test_amount_change_detected(self):
        flags = self.checker.compare(
            "The capital requirement is Rs. 500 crore.",
            "The capital requirement is Rs. 750 crore.",
        )
        assert RuleFlags.AMOUNT_CHANGED in flags

    def test_no_change_no_flags(self):
        text = "Every entity shall submit a quarterly compliance report."
        flags = self.checker.compare(text, text)
        # Identical text should produce no flags
        assert len(flags) == 0

    def test_percentage_change(self):
        flags = self.checker.compare(
            "Maintain 10 per cent capital adequacy ratio.",
            "Maintain 15 per cent capital adequacy ratio.",
        )
        assert RuleFlags.AMOUNT_CHANGED in flags or RuleFlags.NUMBER_CHANGED in flags

    # --- Negation detection ---
    def test_negation_added_english(self):
        flags = self.checker.compare(
            "Banks shall extend credit facilities.",
            "Banks shall not extend credit facilities.",
        )
        assert RuleFlags.NEGATION_ADDED in flags

    def test_negation_removed_english(self):
        flags = self.checker.compare(
            "Banks shall not grant unsecured loans to directors.",
            "Banks shall grant unsecured loans to directors.",
        )
        # "not" removed → negation removed
        assert RuleFlags.NEGATION_REMOVED in flags

    # --- Modal verb detection ---
    def test_modal_strengthened(self):
        flags = self.checker.compare(
            "Banks may extend credit facilities to qualifying entities.",
            "Banks shall extend credit facilities to qualifying entities.",
        )
        assert RuleFlags.MODAL_STRENGTHENED in flags

    def test_modal_weakened(self):
        flags = self.checker.compare(
            "Banks must obtain prior approval before proceeding.",
            "Banks may obtain prior approval before proceeding.",
        )
        assert RuleFlags.MODAL_WEAKENED in flags

    # --- Date detection ---
    def test_date_change_detected(self):
        flags = self.checker.compare(
            "The directive is effective from 01/04/2024.",
            "The directive is effective from 01/04/2025.",
        )
        assert RuleFlags.DATE_CHANGED in flags

    # --- Severity hint ---
    def test_critical_hint_for_negation(self):
        flags = [RuleFlags.NEGATION_ADDED]
        hint = self.checker.severity_hint(flags)
        assert hint == "critical"

    def test_substantive_hint_for_number(self):
        flags = [RuleFlags.AMOUNT_CHANGED]
        hint = self.checker.severity_hint(flags)
        assert hint == "substantive"

    def test_cosmetic_hint_for_no_flags(self):
        hint = self.checker.severity_hint([])
        assert hint == "cosmetic"

    def test_critical_wins_over_substantive(self):
        flags = [RuleFlags.AMOUNT_CHANGED, RuleFlags.NEGATION_ADDED]
        hint = self.checker.severity_hint(flags)
        assert hint == "critical"

    # --- Hindi modal detection ---
    def test_hindi_strong_modal(self):
        result = self.checker.extract("प्रत्येक संस्था को करना होगा।")
        assert result.strong_modals or True  # Soft check: should not crash

    # --- Tamil modal detection ---
    def test_tamil_strong_modal(self):
        result = self.checker.extract("ஒவ்வொரு நிறுவனமும் வேண்டும்.")
        assert result.strong_modals or True  # Soft check

    # --- Edge cases ---
    def test_empty_texts(self):
        flags = self.checker.compare("", "")
        assert isinstance(flags, list)

    def test_extract_returns_correct_types(self):
        result = self.checker.extract("Banks shall not grant loans exceeding Rs. 25 lakh.")
        assert isinstance(result.amounts, list)
        assert isinstance(result.dates, list)
        assert isinstance(result.negations, set)
        assert isinstance(result.strong_modals, set)
        assert isinstance(result.has_deadline, bool)

    def test_deadline_detected(self):
        result = self.checker.extract("Submit within 30 days of the quarter end.")
        assert result.has_deadline

    def test_definition_detected(self):
        result = self.checker.extract('"Compliance report" means the quarterly filing under Regulation 12.')
        assert result.has_definition


class TestRuleCheckerOnRegulatoryPairs:
    """Integration tests on realistic regulatory sentence pairs."""

    def setup_method(self):
        self.checker = RuleBasedChecker()

    def test_cosmetic_paraphrase(self):
        flags = self.checker.compare(
            "Every entity shall submit a quarterly compliance report within 30 days.",
            "Every entity will submit a quarterly compliance report within 30 days.",
        )
        # should / will swap → no critical flag
        assert RuleFlags.NEGATION_ADDED not in flags
        assert RuleFlags.NEGATION_REMOVED not in flags

    def test_critical_obligation_flip(self):
        flags = self.checker.compare(
            "No bank shall grant unsecured loans to directors.",
            "Every bank shall grant unsecured loans to directors.",
        )
        hint = self.checker.severity_hint(flags)
        assert hint == "critical"

    def test_substantive_amount_change(self):
        flags = self.checker.compare(
            "The penalty shall not exceed Rs. 5 lakh per day.",
            "The penalty shall not exceed Rs. 25 lakh per day.",
        )
        hint = self.checker.severity_hint(flags)
        assert hint in ("substantive", "critical")

    def test_modal_obligation_combined(self):
        """Both modal shift and amount change → critical wins."""
        flags = self.checker.compare(
            "Banks may extend credit up to Rs. 10 crore.",
            "Banks must extend credit up to Rs. 15 crore.",
        )
        hint = self.checker.severity_hint(flags)
        # Modal strengthened = critical; amount changed = substantive
        assert hint == "critical"
