"""
Multi-format document readers supporting PDF, DOCX, and TXT files with metadata extraction.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import re

from ..models import Document, SentenceItem
from .splitter import LegalSentenceSplitter


class DocumentReader:
    """Ingests legal/regulatory documents across PDF, DOCX, and TXT formats."""

    def __init__(self, splitter: Optional[LegalSentenceSplitter] = None):
        self.splitter = splitter or LegalSentenceSplitter()

    def read(self, file_path: Union[str, Path], doc_id: Optional[str] = None, version: str = "v1") -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_id = doc_id or path.stem
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._read_pdf(path, doc_id, version)
        elif ext in [".docx", ".doc"]:
            return self._read_docx(path, doc_id, version)
        elif ext in [".txt", ".md", ".json"]:
            return self._read_txt(path, doc_id, version)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .docx, .txt")

    def _read_pdf(self, path: Path, doc_id: str, version: str) -> Document:
        """Reads PDF using pdfplumber with PyMuPDF (fitz) fallback."""
        raw_text_parts = []
        page_texts = []
        metadata: Dict[str, Any] = {"pages": 0, "source": str(path)}

        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                metadata["pages"] = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text(layout=False) or ""
                    clean_text = self._normalize_pdf_text(page_text)
                    page_texts.append((i + 1, clean_text))
                    raw_text_parts.append(clean_text)
        except Exception as e_pdfplumber:
            # Fallback to PyMuPDF
            try:
                import fitz
                doc = fitz.open(path)
                metadata["pages"] = len(doc)
                for i in range(len(doc)):
                    page_text = doc[i].get_text()
                    clean_text = self._normalize_pdf_text(page_text)
                    page_texts.append((i + 1, clean_text))
                    raw_text_parts.append(clean_text)
                doc.close()
            except Exception as e_fitz:
                raise RuntimeError(f"Failed to read PDF '{path}' with both pdfplumber and pymupdf: {e_pdfplumber}; {e_fitz}")

        full_text = "\n\n".join(raw_text_parts)
        sentences = self.splitter.split_document(full_text, doc_id=doc_id)

        return Document(
            doc_id=doc_id,
            version=version,
            filename=path.name,
            file_type="pdf",
            raw_text=full_text,
            sentences=sentences,
            metadata=metadata
        )

    def _read_docx(self, path: Path, doc_id: str, version: str) -> Document:
        """Reads Microsoft Word DOCX files."""
        try:
            import docx
            doc = docx.Document(path)
            paragraphs = []
            for p in doc.paragraphs:
                text = p.text.strip()
                if text:
                    paragraphs.append(text)

            # Also ingest table cells if present
            for table in doc.tables:
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        paragraphs.append(" | ".join(row_texts))

            full_text = "\n\n".join(paragraphs)
            sentences = self.splitter.split_document(full_text, doc_id=doc_id)

            return Document(
                doc_id=doc_id,
                version=version,
                filename=path.name,
                file_type="docx",
                raw_text=full_text,
                sentences=sentences,
                metadata={"source": str(path), "paragraphs_count": len(paragraphs)}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read DOCX file '{path}': {e}")

    def _read_txt(self, path: Path, doc_id: str, version: str) -> Document:
        """Reads plain text / markdown files with UTF-8 encoding."""
        try:
            full_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            full_text = path.read_text(encoding="latin-1")

        sentences = self.splitter.split_document(full_text, doc_id=doc_id)

        return Document(
            doc_id=doc_id,
            version=version,
            filename=path.name,
            file_type="txt",
            raw_text=full_text,
            sentences=sentences,
            metadata={"source": str(path), "char_count": len(full_text)}
        )

    def _normalize_pdf_text(self, text: str) -> str:
        """Clean hyphenated line wraps and excessive whitespace in extracted PDF text."""
        # Fix hyphenated words at line endings e.g. "regula-\ntion" -> "regulation"
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Normalize multiple spaces into single space, keep newlines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(lines)


def read_document(file_path: Union[str, Path], doc_id: Optional[str] = None, version: str = "v1") -> Document:
    """Convenience function to parse any supported document."""
    reader = DocumentReader()
    return reader.read(file_path, doc_id=doc_id, version=version)
