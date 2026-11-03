"""
Unit tests for multi-format document readers.
"""

import pytest
from pathlib import Path
from semdiff.ingestion.readers import DocumentReader


def test_read_txt(tmp_path: Path):
    txt_file = tmp_path / "sample_circular.txt"
    txt_file.write_text(
        "Section 1. Short Title.\n\n"
        "This circular may be called the Banking Regulation Circular 2026. "
        "It shall come into force w.e.f. 1st May 2026.",
        encoding="utf-8"
    )

    reader = DocumentReader()
    doc = reader.read(txt_file, doc_id="circ_001", version="v1")

    assert doc.doc_id == "circ_001"
    assert doc.file_type == "txt"
    assert len(doc.sentences) >= 2
    assert "Banking Regulation Circular 2026" in doc.raw_text


def test_read_docx(tmp_path: Path):
    import docx
    docx_file = tmp_path / "sample_regulation.docx"
    doc_obj = docx.Document()
    doc_obj.add_heading("Regulation 5: Capital Adequacy", level=1)
    doc_obj.add_paragraph("Banks must maintain a minimum capital adequacy ratio of 11.5%.")
    doc_obj.add_paragraph("Failure to comply will attract penalties under Section 46.")
    doc_obj.save(docx_file)

    reader = DocumentReader()
    doc = reader.read(docx_file, doc_id="reg_005", version="v2")

    assert doc.doc_id == "reg_005"
    assert doc.file_type == "docx"
    assert len(doc.sentences) >= 2
    assert any("11.5%" in s.text for s in doc.sentences)


def test_unsupported_format(tmp_path: Path):
    fake_file = tmp_path / "unsupported.xyz"
    fake_file.write_text("Hello world")

    reader = DocumentReader()
    with pytest.raises(ValueError, match="Unsupported file format"):
        reader.read(fake_file)
