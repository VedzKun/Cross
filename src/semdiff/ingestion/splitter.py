"""
Indic-aware and English legal sentence splitter with clause hierarchy extraction.
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from ..models import SentenceItem


class LegalSentenceSplitter:
    """
    Sentence segmenter specialized for legal and regulatory documents in English,
    Hindi, and Tamil. Preserves abbreviations, decimals, and extracts hierarchical
    clause numbers.
    """

    # Common legal and generic abbreviations in English, Hindi, and Tamil
    ENGLISH_ABBREVIATIONS = {
        "sec.", "secs.", "art.", "arts.", "reg.", "regs.", "no.", "nos.",
        "w.e.f.", "dt.", "para.", "paras.", "v.", "vs.", "viz.", "i.e.", "e.g.",
        "approx.", "rs.", "cr.", "ltd.", "pvt.", "govt.", "hon'ble.", "hon.",
        "dr.", "shri.", "smt.", "mr.", "mrs.", "ms.", "co.", "inc.", "corp.",
        "al.", "etc.", "ord.", "cir.", "notif."
    }

    HINDI_ABBREVIATIONS = {
        "सं.", "दि.", "धारा.", "उदा.", "रु.", "श्री.", "डॉ.", "वि.", "क्र."
    }

    TAMIL_ABBREVIATIONS = {
        "எண்.", "நாள்.", "பிரிவு.", "ரூ.", "திரு.", "முனைவர்.", "பக்."
    }

    # Regex patterns for clause headers
    CLAUSE_PATTERNS = [
        # English: "Section 4(1)(a)", "Regulation 3B", "Clause 12."
        re.compile(
            r"^(?:Section|Sec\.|Regulation|Reg\.|Rule|Clause|Article|Art\.)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*[\.:]?)",
            re.IGNORECASE
        ),
        # Hindi: "धारा 4(1)", "विनियम 3ख", "खंड 12."
        re.compile(
            r"^(?:धारा|विनियम|नियम|खंड|अनुच्छेद)\s*([0-9\u0966-\u096F]+[A-Za-z\u0900-\u097F]?(?:\([0-9a-zA-Z\u0900-\u097F]+\))*[\.:]?)",
            re.IGNORECASE
        ),
        # Tamil: "பிரிவு 4(1)", "விதி 3"
        re.compile(
            r"^(?:பிரிவு|விதி|முறைமை|உட்பிரிவு)\s*([0-9\u0BE6-\u0BEF]+[A-Za-z\u0B80-\u0BFF]?(?:\([0-9a-zA-Z\u0B80-\u0BFF]+\))*[\.:]?)",
            re.IGNORECASE
        ),
        # Subclause enumeration: "(a)", "(1)", "(iv)", "(क)" at line start
        re.compile(
            r"^\s*(\([0-9a-zA-ZivxIVX\u0900-\u097F\u0B80-\u0BFF]+\))\s+"
        ),
        # Standalone numeric numbering: "1.", "1.1", "4.2(a)" at line start
        re.compile(
            r"^\s*([0-9]+(?:\.[0-9]+)*[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)\.?\s+"
        ),
    ]

    def __init__(self, split_semicolons_in_lists: bool = True):
        self.split_semicolons_in_lists = split_semicolons_in_lists
        # Combined abbreviations set lowercased
        self.all_abbreviations = (
            self.ENGLISH_ABBREVIATIONS |
            self.HINDI_ABBREVIATIONS |
            self.TAMIL_ABBREVIATIONS
        )

    def detect_language(self, text: str) -> str:
        """Heuristic language detection based on Unicode script block."""
        devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
        tamil_count = len(re.findall(r"[\u0B80-\u0BFF]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))

        if tamil_count > max(devanagari_count, latin_count):
            return "ta"
        if devanagari_count > max(tamil_count, latin_count):
            return "hi"
        return "en"

    def extract_clause_metadata(self, line: str, current_parent_clause: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts clause ID and section title from a line of text if it contains header structure.
        Returns: (clause_id, section_title)
        """
        trimmed = line.strip()
        for pat in self.CLAUSE_PATTERNS:
            m = pat.match(trimmed)
            if m:
                extracted = m.group(1).rstrip(".:")
                if extracted.startswith("(") and current_parent_clause:
                    # If parent already ends with a sibling subclause like "(a)", replace the sibling
                    # e.g. parent "4(1)(a)" + child "(b)" -> "4(1)(b)"
                    sibling_match = re.search(r"\([0-9a-zA-ZivxIVX\u0900-\u097F\u0B80-\u0BFF]+\)$", current_parent_clause)
                    if sibling_match:
                        base_parent = current_parent_clause[:sibling_match.start()]
                        full_clause = f"{base_parent}{extracted}"
                    else:
                        full_clause = f"{current_parent_clause}{extracted}"
                else:
                    full_clause = extracted

                # Check if remainder of line might be a section title
                remainder = trimmed[m.end():].strip().lstrip(":- ")
                title = remainder if len(remainder) > 0 and len(remainder) < 120 else None
                return full_clause, title

        return current_parent_clause, None

    def _protect_tokens(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replaces abbreviations, decimals, and URLs with placeholders to prevent
        erroneous splitting at internal periods.
        """
        replacements: Dict[str, str] = {}
        placeholder_idx = 0

        # Protect decimals and percentages: e.g. 12.5%, 3.14, Rs. 50.50
        def replace_decimal(match: re.Match) -> str:
            nonlocal placeholder_idx
            token = match.group(0)
            key = f"__NUM_{placeholder_idx}__"
            placeholder_idx += 1
            replacements[key] = token
            return key

        text = re.sub(r"\b\d+\.\d+(?:%|[a-zA-Z]+)?\b", replace_decimal, text)

        # Protect known abbreviations
        for abbr in sorted(self.all_abbreviations, key=len, reverse=True):
            escaped = re.escape(abbr)
            # Use raw string concatenation to avoid f-string curly brace conflicts
            pat_str = r"(^|[\s\(\[\{])" + escaped + r"(?=[\s\)\],;:]|$)"
            pattern = re.compile(pat_str, flags=re.IGNORECASE)

            def make_sub(match: re.Match) -> str:
                nonlocal placeholder_idx
                prefix = match.group(1)
                key = f"__ABBR_{placeholder_idx}__"
                placeholder_idx += 1
                replacements[key] = match.group(0)[len(prefix):]
                return prefix + key

            text = pattern.sub(make_sub, text)

        # Protect section/rule references like "Sec. 4(1)" or "w.e.f."
        text = re.sub(r"\bw\.e\.f\.", "__WEF_PROTECT__", text, flags=re.IGNORECASE)
        replacements["__WEF_PROTECT__"] = "w.e.f."

        return text, replacements

    def _restore_tokens(self, text: str, replacements: Dict[str, str]) -> str:
        """Restores protected placeholders back to original text."""
        for key, val in replacements.items():
            text = text.replace(key, val)
        return text

    def split_into_raw_sentences(self, paragraph: str) -> List[str]:
        """
        Splits a single paragraph into raw sentences respecting:
        - Indic Danda '।' (U+0964) and Double Danda '॥' (U+0965)
        - Standard periods '.' (excluding protected abbreviations/numbers)
        - Question mark '?' and exclamation '!'
        - Sub-clause semicolons ';' followed by enumerations
        """
        if not paragraph.strip():
            return []

        protected, replacements = self._protect_tokens(paragraph)
        DELIM = "___SENT_SPLIT_DELIM___"

        # 1. Hindi danda or double danda
        protected = re.sub(r"([।॥]+)(\s*|$)", rf"\1{DELIM}", protected)

        # 2. Exclamation or question marks
        protected = re.sub(r"([\?!]+)(\s+|$)", rf"\1{DELIM}", protected)

        # 3. Standard period followed by whitespace and capital/Indic character, quote, or EOF
        period_pattern = re.compile(
            r"(\.)(?=\s+[\"\'A-Z\u0900-\u097F\u0B80-\u0BFF\(\d]|\s*$)"
        )
        protected = period_pattern.sub(rf"\1{DELIM}", protected)

        # 4. Semicolon before subclause list marker if enabled
        if self.split_semicolons_in_lists:
            semicolon_pattern = re.compile(r"(;)(?=\s*\([a-zA-Z0-9]+\)|\s*\n)")
            protected = semicolon_pattern.sub(rf"\1{DELIM}", protected)

        parts = protected.split(DELIM)
        raw_sentences: List[str] = []

        for p in parts:
            clean = self._restore_tokens(p.strip(), replacements)
            if clean:
                raw_sentences.append(clean)

        return raw_sentences

    def split_document(self, text: str, doc_id: str = "") -> List[SentenceItem]:
        """
        Splits a full document text into a sequence of SentenceItem objects,
        maintaining document structure, clause IDs, and character offsets.
        """
        lines = text.splitlines(keepends=True)
        sentences: List[SentenceItem] = []

        current_clause_id: Optional[str] = None
        current_section_title: Optional[str] = None
        global_char_offset = 0
        sentence_idx = 0

        current_block = ""
        block_start_offset = 0

        for line in lines:
            line_len = len(line)
            stripped = line.strip()

            if not stripped:
                # Blank line indicates paragraph boundary
                if current_block.strip():
                    raw_sents = self.split_into_raw_sentences(current_block)
                    running_offset = block_start_offset
                    for s_text in raw_sents:
                        s_clean = s_text.strip()
                        if not s_clean:
                            continue
                        lang = self.detect_language(s_clean)
                        item = SentenceItem(
                            id=f"{doc_id}_s{sentence_idx:04d}",
                            text=s_clean,
                            doc_id=doc_id,
                            index=sentence_idx,
                            clause_id=current_clause_id,
                            section_title=current_section_title,
                            char_start=running_offset,
                            char_end=running_offset + len(s_clean),
                            language=lang
                        )
                        sentences.append(item)
                        sentence_idx += 1
                        running_offset += len(s_text)
                    current_block = ""
                global_char_offset += line_len
                continue

            # Check if this line introduces a new clause/section header
            new_clause, new_title = self.extract_clause_metadata(stripped, current_clause_id)
            if new_clause != current_clause_id or new_title:
                # Flush previous block
                if current_block.strip():
                    raw_sents = self.split_into_raw_sentences(current_block)
                    running_offset = block_start_offset
                    for s_text in raw_sents:
                        s_clean = s_text.strip()
                        if not s_clean:
                            continue
                        lang = self.detect_language(s_clean)
                        item = SentenceItem(
                            id=f"{doc_id}_s{sentence_idx:04d}",
                            text=s_clean,
                            doc_id=doc_id,
                            index=sentence_idx,
                            clause_id=current_clause_id,
                            section_title=current_section_title,
                            char_start=running_offset,
                            char_end=running_offset + len(s_clean),
                            language=lang
                        )
                        sentences.append(item)
                        sentence_idx += 1
                        running_offset += len(s_text)
                    current_block = ""

                current_clause_id = new_clause
                if new_title:
                    current_section_title = new_title
                block_start_offset = global_char_offset

            if not current_block:
                block_start_offset = global_char_offset
            current_block += line
            global_char_offset += line_len

        # Flush final block
        if current_block.strip():
            raw_sents = self.split_into_raw_sentences(current_block)
            running_offset = block_start_offset
            for s_text in raw_sents:
                s_clean = s_text.strip()
                if not s_clean:
                    continue
                lang = self.detect_language(s_clean)
                item = SentenceItem(
                    id=f"{doc_id}_s{sentence_idx:04d}",
                    text=s_clean,
                    doc_id=doc_id,
                    index=sentence_idx,
                    clause_id=current_clause_id,
                    section_title=current_section_title,
                    char_start=running_offset,
                    char_end=running_offset + len(s_clean),
                    language=lang
                )
                sentences.append(item)
                sentence_idx += 1
                running_offset += len(s_text)

        return sentences


def split_legal_text(text: str, doc_id: str = "") -> List[SentenceItem]:
    """Utility wrapper for LegalSentenceSplitter."""
    splitter = LegalSentenceSplitter()
    return splitter.split_document(text, doc_id=doc_id)
