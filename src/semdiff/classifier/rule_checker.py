"""
Rule-based feature extractor for regulatory text change classification.

Detects: numbers/amounts/dates changed, modal verb shifts (may/shall/must),
negation flips, and other high-signal linguistic patterns across English,
Hindi, and Tamil.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple


# -----------------------------------------------------------------------
# Feature flag constants
# -----------------------------------------------------------------------
class RuleFlags:
    NUMBER_CHANGED = "number_changed"
    DATE_CHANGED = "date_changed"
    AMOUNT_CHANGED = "amount_changed"
    MODAL_STRENGTHENED = "modal_strengthened"   # may → shall/must
    MODAL_WEAKENED = "modal_weakened"           # shall/must → may
    NEGATION_ADDED = "negation_added"
    NEGATION_REMOVED = "negation_removed"
    DEADLINE_CHANGED = "deadline_changed"
    DEFINITION_CHANGED = "definition_changed"
    OBLIGATION_CHANGED = "obligation_changed"

    @classmethod
    def all(cls) -> List[str]:
        return [
            v for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str)
        ]


# -----------------------------------------------------------------------
# Regex banks
# -----------------------------------------------------------------------

# Monetary amounts: Rs., ₹, USD, crore, lakh, etc.
_AMOUNT_PATTERNS = [
    r"(?:Rs\.?|₹|USD|INR)\s*[\d,]+(?:\.\d+)?(?:\s*(?:crore|lakh|thousand|million|billion))?",
    r"\b[\d,]+(?:\.\d+)?\s*(?:crore|lakh|thousand|cr\.?)\b",
    r"\b[\d,]+(?:\.\d+)?\s*%",
]

# Dates: various formats
_DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",           # 01/04/2024, 1-4-24
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",                        # ISO date
    # Hindi months
    r"\b\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+\d{4}\b",
]

# Generic numbers (standalone integers/decimals)
_NUMBER_PATTERNS = [
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
]

# Modal verbs — English
_STRONG_MODALS_EN = {"shall", "must", "will", "required to", "obliged to", "obligated to"}
_WEAK_MODALS_EN = {"may", "can", "could", "should", "might"}

# Modal verbs — Hindi
_STRONG_MODALS_HI = {"चाहिए", "होगा", "होगी", "होंगे", "करना होगा", "अनिवार्य"}
_WEAK_MODALS_HI = {"सकता", "सकती", "सकते", "हो सकता"}

# Modal verbs — Tamil
_STRONG_MODALS_TA = {"வேண்டும்", "கட்டாயம்", "அவசியம்"}
_WEAK_MODALS_TA = {"இயலும்", "முடியும்", "கூடும்"}

# Negations — English
_NEG_EN = {
    "not", "no", "never", "neither", "nor", "without", "unless",
    "except", "nothing", "nobody", "nowhere",
}
# Negations — Hindi (prefix नहीं or न)
_NEG_HI_PATTERN = re.compile(r"\bनहीं\b|\bन\b|\bमत\b")
# Negations — Tamil
_NEG_TA_PATTERN = re.compile(r"\bஇல்லை\b|\bமாட்டேன்\b|\bமாட்டோம்\b|\bவேண்டாம்\b")

# Definition signals
_DEF_EN_PATTERN = re.compile(r'\b(?:means?|refers?\s+to|is\s+defined\s+as|")\b', re.IGNORECASE)
_DEF_HI_PATTERN = re.compile(r'\b(?:का अर्थ|से तात्पर्य|परिभाषित)\b')

# Deadline words
_DEADLINE_EN = {"within", "by", "before", "deadline", "no later than", "not later than",
                "days", "months", "years", "business days", "working days"}
_DEADLINE_HI = {"दिन", "माह", "वर्ष", "तक", "के भीतर"}
_DEADLINE_TA = {"நாள்", "மாதம்", "ஆண்டு", "உள்ளே", "முன்பு"}


# -----------------------------------------------------------------------
# Extractor
# -----------------------------------------------------------------------

@dataclass
class ExtractionResult:
    amounts: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    strong_modals: Set[str] = field(default_factory=set)
    weak_modals: Set[str] = field(default_factory=set)
    negations: Set[str] = field(default_factory=set)
    has_definition: bool = False
    has_deadline: bool = False

    @property
    def modal_strength(self) -> str:
        if self.strong_modals and not self.weak_modals:
            return "strong"
        elif self.weak_modals and not self.strong_modals:
            return "weak"
        elif self.strong_modals and self.weak_modals:
            return "mixed"
        return "none"


class RuleBasedChecker:
    """
    Extracts rule-based features from sentence pair (text_a, text_b)
    and returns a list of triggered RuleFlags strings as auxiliary signal
    for the classifier.
    """

    def __init__(self):
        self._amount_re = re.compile("|".join(_AMOUNT_PATTERNS), re.IGNORECASE | re.UNICODE)
        self._date_re = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE | re.UNICODE)
        self._number_re = re.compile("|".join(_NUMBER_PATTERNS), re.UNICODE)

    def extract(self, text: str) -> ExtractionResult:
        """Extract all relevant features from a single text."""
        result = ExtractionResult()
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower, re.UNICODE))

        # Amounts (check before numbers to avoid double-counting)
        result.amounts = self._amount_re.findall(text)

        # Dates
        result.dates = self._date_re.findall(text)

        # Numbers (excluding already-matched amounts and dates)
        result.numbers = self._number_re.findall(text)

        # English modals
        result.strong_modals = words & _STRONG_MODALS_EN
        result.weak_modals = words & _WEAK_MODALS_EN

        # Hindi modals
        for m in _STRONG_MODALS_HI:
            if m in text:
                result.strong_modals.add(m)
        for m in _WEAK_MODALS_HI:
            if m in text:
                result.weak_modals.add(m)

        # Tamil modals
        for m in _STRONG_MODALS_TA:
            if m in text:
                result.strong_modals.add(m)
        for m in _WEAK_MODALS_TA:
            if m in text:
                result.weak_modals.add(m)

        # Negations — English
        result.negations = words & _NEG_EN
        # Negations — Hindi
        if _NEG_HI_PATTERN.search(text):
            result.negations.add("नहीं")
        # Negations — Tamil
        if _NEG_TA_PATTERN.search(text):
            result.negations.add("இல்லை")

        # Definitions
        result.has_definition = bool(
            _DEF_EN_PATTERN.search(text) or _DEF_HI_PATTERN.search(text)
        )

        # Deadlines
        deadline_words = words & (_DEADLINE_EN | _DEADLINE_HI | _DEADLINE_TA)
        result.has_deadline = bool(deadline_words)

        return result

    def compare(self, text_a: str, text_b: str) -> List[str]:
        """
        Compare two texts and return list of triggered rule flags.
        This is the main entry point used by the classifier pipeline.
        """
        r_a = self.extract(text_a)
        r_b = self.extract(text_b)
        flags: List[str] = []

        # --- Amount changes ---
        set_a = set(a.strip().lower() for a in r_a.amounts)
        set_b = set(b.strip().lower() for b in r_b.amounts)
        if set_a != set_b and (set_a or set_b):
            flags.append(RuleFlags.AMOUNT_CHANGED)

        # --- Date changes ---
        dates_a = set(d.strip().lower() for d in r_a.dates)
        dates_b = set(d.strip().lower() for d in r_b.dates)
        if dates_a != dates_b and (dates_a or dates_b):
            flags.append(RuleFlags.DATE_CHANGED)
            if _has_deadline_context(text_a) or _has_deadline_context(text_b):
                flags.append(RuleFlags.DEADLINE_CHANGED)

        # --- Number changes (pure numeric, not already amounts/dates) ---
        nums_a = _filter_numbers(r_a.numbers, r_a.amounts, r_a.dates)
        nums_b = _filter_numbers(r_b.numbers, r_b.amounts, r_b.dates)
        if nums_a != nums_b and (nums_a or nums_b):
            flags.append(RuleFlags.NUMBER_CHANGED)

        # --- Modal strength shift ---
        strength_a = r_a.modal_strength
        strength_b = r_b.modal_strength
        if strength_a in ("weak", "none") and strength_b == "strong":
            flags.append(RuleFlags.MODAL_STRENGTHENED)
        elif strength_a == "strong" and strength_b in ("weak", "none"):
            flags.append(RuleFlags.MODAL_WEAKENED)

        # --- Negation changes ---
        had_neg_a = bool(r_a.negations)
        had_neg_b = bool(r_b.negations)
        if not had_neg_a and had_neg_b:
            flags.append(RuleFlags.NEGATION_ADDED)
        elif had_neg_a and not had_neg_b:
            flags.append(RuleFlags.NEGATION_REMOVED)

        # --- Obligation changes ---
        if flags and (RuleFlags.MODAL_STRENGTHENED in flags or RuleFlags.MODAL_WEAKENED in flags):
            flags.append(RuleFlags.OBLIGATION_CHANGED)

        # --- Definition changes ---
        if r_a.has_definition != r_b.has_definition:
            flags.append(RuleFlags.DEFINITION_CHANGED)

        return flags

    def severity_hint(self, flags: List[str]) -> str:
        """
        Translate rule flags into a severity hint: 'critical', 'substantive', or 'cosmetic'.
        This is an auxiliary signal, not the final label.
        """
        critical_flags = {
            RuleFlags.NEGATION_ADDED,
            RuleFlags.NEGATION_REMOVED,
            RuleFlags.MODAL_STRENGTHENED,
            RuleFlags.MODAL_WEAKENED,
            RuleFlags.OBLIGATION_CHANGED,
        }
        substantive_flags = {
            RuleFlags.NUMBER_CHANGED,
            RuleFlags.AMOUNT_CHANGED,
            RuleFlags.DATE_CHANGED,
            RuleFlags.DEADLINE_CHANGED,
            RuleFlags.DEFINITION_CHANGED,
        }
        flag_set = set(flags)
        if flag_set & critical_flags:
            return "critical"
        elif flag_set & substantive_flags:
            return "substantive"
        return "cosmetic"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _has_deadline_context(text: str) -> bool:
    return bool(re.search(
        r"\b(?:within|by|before|deadline|no later than|not later than|days?|months?|years?)\b",
        text,
        re.IGNORECASE
    ))


def _filter_numbers(numbers: List[str], amounts: List[str], dates: List[str]) -> Set[str]:
    """Remove number tokens that are already covered by amounts or dates."""
    all_covered = " ".join(amounts + dates).lower()
    result = set()
    for n in numbers:
        if n.replace(",", "") not in all_covered:
            result.add(n)
    return result

