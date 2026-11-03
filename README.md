# Cross-Lingual Semantic Diff for Regulatory Documents (`semdiff`)

A lightweight, **self-hostable**, CPU-first tool that compares versions of legal/regulatory documents **by meaning** — not by character — and classifies each change as `cosmetic`, `substantive`, or `critical`. Works across **English, Hindi, and Tamil**, even when versions are in different languages.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt
pip install -e .

# 2. Compare two documents
semdiff compare v1.txt v2.txt

# 3. Multi-version chain (v1 → v2 → v3)
semdiff compare v1.pdf v2.pdf v3.pdf --output reports/chain.html

# 4. Generate synthetic training data
semdiff dataset

# 5. Run all tests
python -m pytest -v
```

---

## 4-Phase Roadmap

| Phase | Description | Status |
|:------|:-----------|:-------|
| **Phase 1** | Ingestion & Indic Segmentation — PDF/DOCX/TXT reader, Indic/English legal splitter, LaBSE embedding cache | ✅ Done |
| **Phase 2** | Semantic Alignment — Needleman-Wunsch DP (1-1, split, merge, insert, delete) + multi-version chain | ✅ Done |
| **Phase 3** | Dataset & Severity Classifier — Synthetic RBI/SEBI generator, DistilBERT + rule-based classifier | ✅ Done |
| **Phase 4** | Output, CLI, UI & Evaluation — HTML/JSON diff, CLI, FastAPI, Streamlit, Dockerfile, full benchmarks | ✅ Done |

---

## Repository Structure

```
Cross/
├── src/semdiff/
│   ├── ingestion/
│   │   ├── readers.py          # PDF / DOCX / TXT ingestion
│   │   └── splitter.py         # Indic-aware legal sentence splitter
│   ├── embedding/
│   │   ├── labse.py            # LaBSE embedder (sentence-transformers)
│   │   └── cache.py            # SQLite disk cache
│   ├── alignment/
│   │   ├── aligner.py          # Needleman-Wunsch DP aligner
│   │   └── chain.py            # Multi-version amendment chain
│   ├── classifier/
│   │   ├── rule_checker.py     # Rule-based features (numbers, modals, negations)
│   │   ├── distilbert_classifier.py  # DistilBERT fine-tuning wrapper
│   │   └── pipeline.py         # Combined rule + model classification
│   ├── reporting/
│   │   ├── html_report.py      # Self-contained HTML diff report
│   │   └── json_export.py      # JSON export
│   ├── cli.py                  # Typer CLI (semdiff compare / train / dataset)
│   ├── api.py                  # FastAPI REST API
│   ├── ui.py                   # Streamlit web UI
│   ├── config.py               # Centralized settings
│   └── models.py               # Pydantic data models
├── data/
│   ├── scripts/
│   │   ├── build_dataset.py    # Synthetic regulatory pair generator
│   │   └── split_dataset.py    # Document-level train/val/test split
│   ├── synthetic/              # Generated pairs + labeling template
│   └── processed/              # train.csv / val.csv / test.csv
├── evaluation/
│   ├── eval_alignment.py       # Alignment precision / recall / F1
│   ├── eval_classifier.py      # Classification metrics + confusion matrix
│   ├── eval_baselines.py       # difflib / TF-IDF / LaBSE-only baselines
│   ├── eval_crosslingual.py    # EN ↔ HI / EN ↔ TA cross-lingual test
│   ├── failure_analysis.py     # Per-failure-mode analysis
│   └── efficiency.py           # Latency + memory on CPU
├── tests/
│   ├── test_ingestion.py       # Reader unit tests
│   ├── test_splitter.py        # Indic splitter unit tests
│   ├── test_embedding.py       # Cache + similarity unit tests
│   ├── test_alignment.py       # DP aligner unit tests (14 tests)
│   └── test_classifier.py      # Rule checker unit tests (18 tests)
├── reports/                    # Generated HTML reports
├── models/classifier/          # Fine-tuned DistilBERT checkpoint
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Pipeline

```
PDF/DOCX/TXT
     │
     ▼
[1] Ingestion          readers.py + splitter.py
     │   Indic danda (।) aware sentence splitting
     │   Clause/section metadata extraction
     │   Protection of abbreviations, decimals, legal refs
     ▼
[2] Embedding          labse.py + cache.py
     │   LaBSE (sentence-transformers/LaBSE)
     │   Normalized 768-dim vectors, SQLite disk cache
     ▼
[3] Alignment          aligner.py
     │   Cosine similarity matrix
     │   Needleman-Wunsch DP with gap penalties
     │   Resolves: 1-1, 1-many (split), many-1 (merge), insert, delete
     ▼
[4] Classification     pipeline.py
     │   Rule-based flags: numbers, dates, amounts, modals, negations
     │   DistilBERT (distilbert-base-multilingual-cased) fine-tuned
     │   Final label = max(rule_hint, model_label)
     │   Labels: cosmetic / substantive / critical
     ▼
[5] Chain Analysis     chain.py
     │   v1 → v2 → v3 consecutive diff tracking
     │   Per-clause lineage (ClauseTimeline)
     ▼
[6] Output             html_report.py + json_export.py
         Side-by-side HTML with color-coded severity + confidence
         JSON export of all pairs, timelines, stats
```

---

## CLI Reference

```bash
# Compare documents (rule_only is fast, no GPU needed)
semdiff compare a.pdf b.pdf
semdiff compare a.pdf b.pdf --output reports/diff.html --json diff.json
semdiff compare a.pdf b.pdf --mode combined --checkpoint models/classifier

# Multi-version chain
semdiff compare v1.txt v2.txt v3.txt

# Fine-tune DistilBERT classifier
semdiff train data/processed/train.csv --val data/processed/val.csv

# Generate synthetic dataset + split
semdiff dataset --cosmetic 150 --substantive 100 --critical 100
```

### Classification Modes

| Mode | Speed | Requires model | When to use |
|:-----|:------|:--------------|:-----------|
| `rule_only` | ~9,000 pairs/s | No | Quick runs, no checkpoint needed |
| `combined` | ~20 pairs/s | Yes (fine-tuned) | Best accuracy |
| `model_only` | ~20 pairs/s | Yes | Ablation study |

---

## FastAPI (REST)

```bash
uvicorn src.semdiff.api:app --host 0.0.0.0 --port 8000
```

```http
POST /compare/text       # JSON body: {text_a, text_b, mode, format}
POST /compare            # multipart/form-data with file_a, file_b
GET  /health
```

---

## Streamlit UI

```bash
streamlit run src/semdiff/ui.py
```

Upload documents or paste text, select mode, view color-coded diff, download HTML/JSON.

---

## Docker

```bash
# Build
docker build -t semdiff .

# Run FastAPI
docker run -p 8000:8000 semdiff

# Run Streamlit
docker run -p 8501:8501 semdiff \
  streamlit run src/semdiff/ui.py --server.port=8501 --server.address=0.0.0.0
```

---

## Training the Classifier

### Step 1 — Generate synthetic data

```bash
python data/scripts/build_dataset.py \
  --cosmetic 150 --substantive 100 --critical 100 \
  --output-dir data/synthetic
```

Produces `pairs.csv` (363 pairs: cosmetic/substantive/critical across EN, HI, TA) and a `labeling_template.csv` for hand annotation.

### Step 2 — Hand-label (optional but recommended)

Open `data/synthetic/labeling_template.csv` in Excel/LibreOffice. Fill in the `label_human` column with: `cosmetic`, `substantive`, or `critical`. Aim for 300-500 rows.

### Step 3 — Split by document

```bash
python data/scripts/split_dataset.py \
  --input data/synthetic/pairs.csv \
  --output-dir data/processed
```

Splits 70/15/15 by **document ID** (not sentence), preventing data leakage. Outputs `train.csv`, `val.csv`, `test.csv`.

### Step 4 — Fine-tune DistilBERT

```bash
semdiff train data/processed/train.csv \
  --val data/processed/val.csv \
  --epochs 5 \
  --batch-size 16
```

Saves checkpoint to `models/classifier/`. Uses inverse-frequency class weights for imbalance.

Input format: `[CLS] sentence_a [SEP] sentence_b` (truncated to 256 tokens).

---

## Evaluation

```bash
# Alignment quality (P/R/F1 against gold alignments)
python -X utf8 evaluation/eval_alignment.py --demo

# Classifier metrics + confusion matrix
python -X utf8 evaluation/eval_classifier.py --demo

# Baseline comparison (difflib / TF-IDF / LaBSE-threshold)
python -X utf8 evaluation/eval_baselines.py --demo

# Cross-lingual EN ↔ HI / EN ↔ TA test
python -X utf8 evaluation/eval_crosslingual.py --demo

# Failure-mode analysis
python -X utf8 evaluation/failure_analysis.py

# CPU latency + memory benchmark
python evaluation/efficiency.py --n-sents 50 --skip-labse
```

### Baseline Results (Demo — rule_only mode)

| Method | Macro F1 | Accuracy | Notes |
|:-------|:--------:|:--------:|:------|
| difflib (char-level) | 0.15 | 0.30 | Misses semantic changes |
| TF-IDF cosine | 0.39 | 0.50 | Better but language-blind |
| LaBSE-only threshold | 0.51 | 0.50 | Good cross-lingual but no severity logic |
| **semdiff (rule_only)** | **0.50** | **0.50** | Fast, no model needed |
| **semdiff (combined)** | _train to see_ | — | Best with fine-tuned checkpoint |

### Cross-Lingual Results (LaBSE, mock embeddings)

| Pair | Avg Cosine Similarity | Pairs ≥ 0.70 |
|:-----|:---------------------:|:------------:|
| EN ↔ HI (5 clauses) | 0.813 | 5/5 |
| EN ↔ TA (5 clauses) | 0.857 | 5/5 |

### Failure Mode Analysis (rule_only)

| Failure Mode | Accuracy |
|:------------|:--------:|
| Negation flip | 66.7% |
| Number / amount change | 100% |
| Long compound sentences | 100% |
| OCR noise (1/l confusion) | 0% ← known limitation |
| Modal shift (may→shall) | 100% |
| Cross-lingual negation | 100% |
| **Overall** | **78.6%** |

### Efficiency (CPU, 4-core laptop)

| Step | Throughput | RAM |
|:-----|:----------:|:---:|
| Ingestion | ~700 sent/s | <1 MB |
| DP alignment (50 sent docs) | ~476 pairs/s | <1 MB |
| Rule checker | ~9,300 pairs/s | <1 MB |
| LaBSE (no cache) | ~30 sent/s | ~2 GB |
| LaBSE (cached) | ~50,000 sent/s | <1 MB |

---

## Label Definitions

| Label | Description | Examples |
|:------|:-----------|:--------|
| `cosmetic` | Same meaning, different wording | shall→will, utilize→use, prior to→before |
| `substantive` | Meaning changes but same obligation direction | Rs. 500→Rs. 750 crore, 30→60 days, quarterly→monthly |
| `critical` | Obligation direction or legal force changes | shall→shall not, may→must, "no person"→"every person" |

---

## Configuration

Edit [`src/semdiff/config.py`](src/semdiff/config.py) or override at runtime:

```python
from semdiff.config import default_settings
default_settings.sim_threshold = 0.60        # alignment sensitivity
default_settings.gap_open_penalty = 0.30     # DP gap penalty
default_settings.split_merge_threshold = 0.60
```

---

## Running Tests

```bash
python -m pytest -v       # 46 tests across all modules
```

Test coverage:
- **Ingestion**: PDF/DOCX/TXT reading, format validation
- **Splitter**: English abbreviations, Hindi danda (।), Tamil text, clause hierarchy, decimals
- **Embedding cache**: SQLite read/write/batch, similarity matrix computation
- **Alignment**: 1-1, insert, delete, empty docs, severity, stats, split/merge, shape mismatch
- **Classifier**: amount/date/number change, negation add/remove, modal shift, severity hints, edge cases

---

## Notes & Known Limitations

1. **LaBSE model download**: First run downloads ~1.9 GB. Subsequent runs use the SQLite cache.
2. **OCR noise**: Rule checker is fooled by `sha11` / `l→1` substitutions. Add a spell-correction preprocessing step for scanned PDFs.
3. **DistilBERT fine-tuning**: Requires hand-labeled data for best results. With `rule_only` mode the tool is useful immediately.
4. **Hindi/Tamil negation**: Rule coverage is heuristic; complete coverage requires a larger lexicon.
5. **Long documents**: DP alignment is O(n×m). For documents >500 sentences, consider chunking by section.
