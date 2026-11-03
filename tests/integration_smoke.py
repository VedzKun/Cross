"""
Integration smoke test: end-to-end pipeline test using mock embeddings.
Tests: ingestion → alignment → classification → HTML report → JSON export → chain analysis.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import tempfile
from pathlib import Path

from semdiff.ingestion.readers import DocumentReader
from semdiff.alignment.aligner import SemanticAligner
from semdiff.alignment.chain import AmendmentChain
from semdiff.classifier.pipeline import ClassificationPipeline
from semdiff.classifier.rule_checker import RuleBasedChecker
from semdiff.reporting.html_report import generate_html_report
from semdiff.reporting.json_export import export_diff_to_json
from semdiff.models import AlignedPair, AlignmentType, SeverityLevel, SentenceItem

print("=" * 60)
print("END-TO-END INTEGRATION SMOKE TEST")
print("=" * 60)

# ─────────────────────────────────────────
# 1. Ingestion
# ─────────────────────────────────────────
print("\n[1] Ingestion...")
text_v1 = """Section 1.1: Every registered entity shall submit a quarterly compliance report within 30 days.

Section 1.2: The minimum capital requirement shall not be less than Rs. 500 crore at all times.

Section 1.3: No person shall engage in foreign exchange transactions without prior approval.

Section 1.4: Banks may extend credit facilities up to Rs. 10 crore under priority sector norms."""

text_v2 = """Section 1.1: Every registered entity must submit a quarterly compliance report within 45 days.

Section 1.2: The minimum capital requirement shall not be less than Rs. 750 crore at all times.

Section 1.3: No person shall engage in foreign exchange transactions without prior approval.

Section 1.5: All payment aggregators shall maintain a nodal account with a scheduled commercial bank."""

with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f1:
    f1.write(text_v1); p1 = Path(f1.name)
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f2:
    f2.write(text_v2); p2 = Path(f2.name)

reader = DocumentReader()
doc_v1 = reader.read(p1, doc_id="rbi_v1", version="v1")
doc_v2 = reader.read(p2, doc_id="rbi_v2", version="v2")
p1.unlink(); p2.unlink()

print(f"  v1: {len(doc_v1.sentences)} sentences")
print(f"  v2: {len(doc_v2.sentences)} sentences")
assert len(doc_v1.sentences) >= 4, "Expected at least 4 sentences in v1"

# ─────────────────────────────────────────
# 2. Mock Embeddings + Alignment
# ─────────────────────────────────────────
print("\n[2] Alignment (mock embeddings)...")
n, m = len(doc_v1.sentences), len(doc_v2.sentences)
np.random.seed(7)
emb_a = np.random.randn(n, 768).astype(np.float32)
emb_b = np.random.randn(m, 768).astype(np.float32)
emb_a /= np.linalg.norm(emb_a, axis=1, keepdims=True)
emb_b /= np.linalg.norm(emb_b, axis=1, keepdims=True)

# Simulate known alignments: sentences 0-2 match (slight drift), sentence 3 deleted, sentence 3 in B is new
for i in range(min(3, n, m)):
    emb_b[i] = emb_a[i] + np.random.randn(768).astype(np.float32) * 0.08
    emb_b[i] /= np.linalg.norm(emb_b[i])

sim_matrix = np.dot(emb_a, emb_b.T)
aligner = SemanticAligner(sim_threshold=0.50)
diff = aligner.align(doc_v1.sentences, doc_v2.sentences, sim_matrix, "rbi_v1", "rbi_v2")

print(f"  Aligned pairs: {len(diff.aligned_pairs)}")
print(f"  Stats: {diff.stats['alignment_type_counts']}")
assert len(diff.aligned_pairs) >= 4

# ─────────────────────────────────────────
# 3. Classification (rule_only)
# ─────────────────────────────────────────
print("\n[3] Classification (rule_only)...")
clf = ClassificationPipeline(mode="rule_only")
diff = clf.classify_diff_result(diff)

sev = diff.stats["severity_counts"]
print(f"  Severity counts: {sev}")
assert "critical" in sev or "substantive" in sev or "cosmetic" in sev

# ─────────────────────────────────────────
# 4. Rule checker spot-check
# ─────────────────────────────────────────
print("\n[4] Rule checker spot-checks...")
checker = RuleBasedChecker()

cases = [
    ("submit within 30 days", "submit within 45 days", {"number_changed"}),
    ("Rs. 500 crore", "Rs. 750 crore", {"amount_changed"}),
    ("banks may extend", "banks shall not extend", {"modal_strengthened", "negation_added"}),
]
for a, b, expected_flags in cases:
    flags = set(checker.compare(a, b))
    matched = flags & expected_flags
    status = "OK" if matched else "MISS"
    print(f"  [{status}] '{a}' vs '{b}' -> flags={flags} (expected any of {expected_flags})")

# ─────────────────────────────────────────
# 5. HTML Report
# ─────────────────────────────────────────
print("\n[5] HTML report generation...")
out_path = Path("reports/integration_test.html")
html = generate_html_report(diff, "RBI Circular v1", "RBI Circular v2", output_path=out_path)
assert "<html" in html.lower()
assert len(html) > 5000
print(f"  Report: {out_path} ({len(html):,} chars)")

# ─────────────────────────────────────────
# 6. JSON export
# ─────────────────────────────────────────
print("\n[6] JSON export...")
import json
json_str = export_diff_to_json(diff)
parsed = json.loads(json_str)
assert "diff" in parsed
assert "aligned_pairs" in parsed["diff"]
print(f"  JSON: {len(json_str):,} chars, {len(parsed['diff']['aligned_pairs'])} pairs")

# ─────────────────────────────────────────
# 7. Amendment Chain
# ─────────────────────────────────────────
print("\n[7] Amendment chain (v1->v2)...")

class MockEmbedder:
    """Embedder that returns precomputed mock embeddings by index."""
    def embed_sentences(self, texts):
        np.random.seed(42)
        vecs = np.random.randn(len(texts), 768).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs
    def compute_similarity_matrix(self, a, b):
        return np.dot(a, b.T)

chain = AmendmentChain(embedder=MockEmbedder())
chain.add_version(doc_v1)
chain.add_version(doc_v2)
diffs = chain.get_diffs()
timelines = chain.get_timelines()
summary = chain.summary()

print(f"  Diffs: {len(diffs)}")
print(f"  Clause timelines: {len(timelines)}")
print(f"  Summary: {summary}")
assert len(diffs) == 1
assert len(timelines) >= 1

print("\n" + "=" * 60)
print("ALL INTEGRATION CHECKS PASSED")
print("=" * 60)

