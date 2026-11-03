"""
Unit tests for Indic-aware and English legal sentence splitter.
"""

import pytest
from semdiff.ingestion.splitter import LegalSentenceSplitter, split_legal_text


def test_english_legal_abbreviations_not_split():
    splitter = LegalSentenceSplitter()
    text = (
        "As per Sec. 4(1) of the Act, every company must submit returns w.e.f. 1st April. "
        "The penalty shall be Rs. 50.50 crores in accordance with Reg. 12(b). "
        "This was upheld in State of Maharashtra v. Union of India."
    )
    sentences = splitter.split_document(text, doc_id="test_en")

    assert len(sentences) == 3
    assert "Sec. 4(1)" in sentences[0].text
    assert "w.e.f." in sentences[0].text
    assert "Rs. 50.50 crores" in sentences[1].text
    assert "v. Union of India" in sentences[2].text
    assert sentences[0].language == "en"


def test_hindi_danda_splitting():
    splitter = LegalSentenceSplitter()
    text = (
        "भारतीय रिज़र्व बैंक ने नए निर्देश जारी किए हैं। "
        "सभी बैंकों को सं. 12 के तहत इन नियमों का पालन करना अनिवार्य होगा। "
        "यह आदेश तत्काल प्रभाव से लागू होगा॥"
    )
    sentences = splitter.split_document(text, doc_id="test_hi")

    assert len(sentences) == 3
    assert sentences[0].language == "hi"
    assert "भारतीय रिज़र्व बैंक ने नए निर्देश जारी किए हैं।" in sentences[0].text
    # Check that Hindi abbreviation सं. did not trigger premature split
    assert "सं. 12" in sentences[1].text
    assert "लागू होगा॥" in sentences[2].text


def test_tamil_legal_text():
    splitter = LegalSentenceSplitter()
    text = (
        "இந்திய ரிசர்வ் வங்கி புதிய வழிகாட்டுதல்களை வெளியிட்டுள்ளது. "
        "பிரிவு 4(1) இன் கீழ் அனைத்து வங்கிகளும் இதனை பின்பற்ற வேண்டும்."
    )
    sentences = splitter.split_document(text, doc_id="test_ta")

    assert len(sentences) == 2
    assert sentences[0].language == "ta"
    assert "இந்திய ரிசர்வ் வங்கி" in sentences[0].text
    assert "பிரிவு 4(1)" in sentences[1].text


def test_clause_hierarchy_extraction():
    splitter = LegalSentenceSplitter()
    text = (
        "Section 4(1). Obligation to Report Transactions:\n"
        "(a) Every reporting entity shall verify client identity.\n"
        "(b) The records shall be maintained for a period of ten years.\n"
    )
    sentences = splitter.split_document(text, doc_id="test_clause")

    assert len(sentences) >= 2
    # Verify clause tracking
    clauses = [s.clause_id for s in sentences if s.clause_id]
    assert any("4(1)" in str(c) for c in clauses)


def test_decimal_and_percentage_preservation():
    splitter = LegalSentenceSplitter()
    text = "The interest rate increased by 2.75% to 8.50% per annum. Next sentence follows."
    sentences = splitter.split_document(text, doc_id="test_num")

    assert len(sentences) == 2
    assert "2.75% to 8.50%" in sentences[0].text
    assert "Next sentence follows." in sentences[1].text
