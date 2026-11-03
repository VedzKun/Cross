"""
FastAPI REST API for semdiff.

Run with:
    uvicorn src.semdiff.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /compare          — Upload two files and get diff JSON
    POST /compare/text     — Compare raw text strings
    GET  /health           — Health check
"""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .ingestion.readers import DocumentReader
from .embedding.labse import LaBSEEmbedder
from .alignment.aligner import SemanticAligner
from .alignment.chain import AmendmentChain
from .classifier.pipeline import ClassificationPipeline
from .reporting.html_report import generate_html_report
from .reporting.json_export import export_diff_to_json
from .config import default_settings

app = FastAPI(
    title="SemDiff API",
    description="Cross-Lingual Semantic Diff for Regulatory Documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_embedder: Optional[LaBSEEmbedder] = None
_reader: Optional[DocumentReader] = None


def get_embedder() -> LaBSEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = LaBSEEmbedder()
    return _embedder


def get_reader() -> DocumentReader:
    global _reader
    if _reader is None:
        _reader = DocumentReader()
    return _reader


class TextCompareRequest(BaseModel):
    text_a: str
    text_b: str
    label_a: str = "Version A"
    label_b: str = "Version B"
    mode: str = "rule_only"
    format: str = "json"  # "json" or "html"


class DiffSummary(BaseModel):
    doc_a: str
    doc_b: str
    total_pairs: int
    unchanged: int
    cosmetic: int
    substantive: int
    critical: int
    insertions: int
    deletions: int
    elapsed_seconds: float


@app.get("/health")
async def health():
    return {"status": "ok", "service": "semdiff"}


@app.post("/compare/text", summary="Compare two raw text strings")
async def compare_text(request: TextCompareRequest):
    """Compare two text strings and return a semantic diff."""
    t0 = time.time()
    reader = get_reader()
    embedder = get_embedder()

    # Write to temp files so we can reuse the file pipeline
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as fa:
        fa.write(request.text_a)
        path_a = Path(fa.name)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as fb:
        fb.write(request.text_b)
        path_b = Path(fb.name)

    try:
        doc_a = reader.read(path_a, doc_id="text_a", version="v1")
        doc_b = reader.read(path_b, doc_id="text_b", version="v2")
    finally:
        path_a.unlink(missing_ok=True)
        path_b.unlink(missing_ok=True)

    aligner = SemanticAligner()
    chain = AmendmentChain(embedder=embedder, aligner=aligner)
    chain.add_version(doc_a)
    chain.add_version(doc_b)
    diffs = chain.get_diffs()
    timelines = chain.get_timelines()

    clf = ClassificationPipeline(mode=request.mode)
    diff = clf.classify_diff_result(diffs[0])

    elapsed = time.time() - t0

    if request.format == "html":
        html = generate_html_report(diff, request.label_a, request.label_b, timelines)
        return HTMLResponse(content=html)

    return JSONResponse(content={
        "diff": diff.model_dump(),
        "timelines": [t.model_dump() for t in timelines],
        "elapsed_seconds": round(elapsed, 2),
    })


@app.post("/compare", summary="Compare two uploaded document files")
async def compare_files(
    file_a: UploadFile = File(..., description="First document version (PDF, DOCX, TXT)"),
    file_b: UploadFile = File(..., description="Second document version"),
    mode: str = Form("rule_only"),
    format: str = Form("json"),
):
    """Upload two regulatory document files and return a semantic diff."""
    t0 = time.time()

    allowed_exts = {".pdf", ".docx", ".txt", ".doc"}

    def _check_ext(upload: UploadFile) -> str:
        name = upload.filename or ""
        ext = Path(name).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Use: {allowed_exts}")
        return ext

    ext_a = _check_ext(file_a)
    ext_b = _check_ext(file_b)

    reader = get_reader()
    embedder = get_embedder()

    # Save uploads to temp files
    with tempfile.NamedTemporaryFile(suffix=ext_a, delete=False) as ta:
        ta.write(await file_a.read())
        path_a = Path(ta.name)
    with tempfile.NamedTemporaryFile(suffix=ext_b, delete=False) as tb:
        tb.write(await file_b.read())
        path_b = Path(tb.name)

    try:
        doc_a = reader.read(path_a, doc_id=file_a.filename or "doc_a", version="v1")
        doc_b = reader.read(path_b, doc_id=file_b.filename or "doc_b", version="v2")
    finally:
        path_a.unlink(missing_ok=True)
        path_b.unlink(missing_ok=True)

    aligner = SemanticAligner()
    chain = AmendmentChain(embedder=embedder, aligner=aligner)
    chain.add_version(doc_a)
    chain.add_version(doc_b)
    diff = chain.get_diffs()[0]
    timelines = chain.get_timelines()

    clf = ClassificationPipeline(mode=mode)
    diff = clf.classify_diff_result(diff)

    elapsed = time.time() - t0

    if format == "html":
        html = generate_html_report(
            diff,
            doc_a_label=file_a.filename or "Version A",
            doc_b_label=file_b.filename or "Version B",
            timelines=timelines,
        )
        return HTMLResponse(content=html)

    return JSONResponse(content={
        "summary": {
            "doc_a": file_a.filename,
            "doc_b": file_b.filename,
            "total_pairs": len(diff.aligned_pairs),
            "elapsed_seconds": round(elapsed, 2),
            **diff.stats.get("severity_counts", {}),
        },
        "diff": diff.model_dump(),
        "timelines": [t.model_dump() for t in timelines],
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

