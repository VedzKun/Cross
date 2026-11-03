# Cross-Lingual Semantic Diff for Regulatory Documents (`semdiff`)
## System Architecture & End-to-End Pipeline Explanation

A technical deep-dive into how `semdiff` ingests, semantically aligns, classifies, and tracks changes across multi-version legal and regulatory documents in **English, Hindi, and Tamil**.

---

## 1. High-Level Pipeline Architecture

```
                       THE COMPLETE PIPELINE FLOW
                       
   Document A (e.g., v1 / EN)           Document B (e.g., v2 / HI / amended)
               │                                         │
               ▼                                         ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ STEP 1: Ingestion & Indic Legal Segmentation                │
   │  • Regex protects decimals, currency (Rs. 50.50 cr), legal  │
   │    abbreviations (Sec., w.e.f., दि., பிரிவு)               │
   │  • Splits on Indic Danda (।), Deergh Viram (॥), & periods  │
   │  • Extracts clause hierarchy ("Section 4(1)(a)", "धारा 5") │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Sentences + Structural Metadata
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ STEP 2: Cross-Lingual Embedding & SQLite Disk Caching       │
   │  • Computes 768-dim normalized dense vectors (LaBSE)        │
   │  • Query/store in SQLite via SHA-256(model:text) hash key   │
   │  • CPU-optimized, zero repeat compute                       │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Dense Embedding Vectors
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ STEP 3: Pairwise Cosine Matrix & DP Sequence Alignment     │
   │  • Matrix M[i, j] = cos_sim(embed_a[i], embed_b[j])         │
   │  • Needleman-Wunsch dynamic programming with gap penalties  │
   │  • Resolves 1-1, 1-many (split), many-1 (merge), ins, del   │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Aligned Sentence Pairs
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ STEP 4: Rule-Augmented Severity Classification              │
   │  • Regex checks: numbers, dates, deadlines, modal shifts    │
   │    (may ↔ shall ↔ must), and negation flips (not / नहीं)   │
   │  • Fine-tuned DistilBERT: [CLS] text_a [SEP] text_b         │
   │  • Decision policy: Max(Rule_Signal, Model_Confidence)      │
   │  • Output labels: {cosmetic, substantive, critical}         │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Classified Pairs + Rule Flags
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ STEP 5: Multi-Version Amendment Chain Tracking (v1→v2→v3)   │
   │  • Links clause IDs across consecutive transitions          │
   │  • Builds ClauseTimeline: active, amended, merged, deleted  │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ STEP 6: Interactive HTML Diff & JSON Serialization          │
   │  • Side-by-side color-coded table (Yellow/Orange/Red)       │
   │  • Confidence gauges, rule badges, and live search bar      │
   └─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Problem Solved

Standard diff engines (`git diff`, `difflib`, word processor redlines) evaluate **character or token edit distance**. In regulatory and legal domains, this fails fundamentally:

1. **False Negatives on Legal Reversals:**
   - *Version A:* `"Banks shall extend credit facilities."`
   - *Version B:* `"Banks shall not extend credit facilities."`
   - Character diff reports **96% identical**. Legally, this is a **180° inversion from permission to prohibition** (`critical`).
2. **False Positives on Paraphrasing and Cross-Lingual Versions:**
   - *Version A (English):* `"Entities must submit compliance reports within 30 days."`
   - *Version B (Hindi):* `"संस्थाओं को 30 दिनों के भीतर अनुपालन रिपोर्ट प्रस्तुत करनी होगी।"`
   - Character diff reports **0% similarity**. Semantically, they are **identical** (`cosmetic` / `unchanged`).

---

## 3. Detailed Step-by-Step Breakdown

### Step 1: Ingestion & Indic Legal Segmentation
**Source Files:** `src/semdiff/ingestion/readers.py`, `src/semdiff/ingestion/splitter.py`

1. **Multi-Format Ingestion:**
   - Ingests `.pdf` (using `pdfplumber` with `PyMuPDF` fallback), Microsoft Word `.docx` (including tables), and plain `.txt`/`.md`.
   - Normalizes line wraps and hyphenated word breaks (e.g. `regula-\ntion` $\to$ `regulation`).
2. **Token Shielding (Protection Phase):**
   - Standard sentence splitters break incorrectly on decimal numbers or legal abbreviations.
   - `semdiff` shields legal numbers (`Rs. 50.50 crores`, `2.75%`), statutory abbreviations (`Sec.`, `w.e.f.`, `Dr.`, `दि.`, `பிரிவு.`), and citations with internal protected tokens (`__NUM_0__`, `__ABBR_1__`).
3. **Indic Boundary Splitting:**
   - Segments text using Hindi Purna Viram Danda (`।` - U+0964), Double Danda (`॥` - U+0965), Latin full-stops, question marks, and sub-clause list semicolons.
4. **Hierarchical Clause & Language Detection:**
   - Identifies structural headings across scripts:
     - English: `Section 4(1)(a)`, `Regulation 3B`
     - Hindi: `धारा 5(1)(क)`, `विनियम 3ख`
     - Tamil: `பிரிவு 6(1)(அ)`, `விதி 3`
   - Automatically detects Unicode script (`en`, `hi`, `ta`) and tags character offsets.

---

### Step 2: Cross-Lingual Embedding & Persistent SQLite Disk Cache
**Source Files:** `src/semdiff/embedding/labse.py`, `src/semdiff/embedding/cache.py`

1. **LaBSE (Language-agnostic BERT Sentence Embedding):**
   - Maps sentences from 109 languages into a shared **768-dimensional L2-normalized vector space**.
   - Identical clauses in English, Hindi, and Tamil map to closely neighboring points.
2. **Zero-Redundancy SQLite Cache:**
   - Calculating neural embeddings on laptop CPUs takes ~30–50 ms per sentence.
   - Embeddings are hashed via `SHA-256(model_name:text)` and stored in `.cache/embeddings.db` using SQLite WAL mode.
   - Cache hits retrieve embeddings in **sub-millisecond** time (~50,000 sentences/sec), enabling instantaneous diffing on re-runs.
3. **Resilient Offline Fallback:**
   - When running in network-restricted environments without pre-downloaded weights, the system automatically uses a deterministic normalized vector fallback, ensuring the CLI/API never crashes.

---

### Step 3: Pairwise Matrix & DP Sequence Alignment
**Source File:** `src/semdiff/alignment/aligner.py`

Once sentence vectors are computed for Document A ($n$ items) and Document B ($m$ items):

1. **Pairwise Cosine Matrix:**
   - Computes an $n \times m$ similarity matrix:
     $$\text{sim}(a_i, b_j) = \mathbf{u}_{a_i} \cdot \mathbf{v}_{b_j} \quad (\in [-1, 1])$$
2. **Needleman-Wunsch Dynamic Programming:**
   - Legal documents follow sequential flow with inserted, deleted, or reordered clauses.
   - A DP matrix is filled using gap-open and gap-extend penalties:
     $$\text{DP}[i, j] = \max \begin{cases} \text{DP}[i-1, j-1] + \text{sim}(a_i, b_j) & \text{(Match)} \\ \text{DP}[i-1, j] - \text{gap\_penalty} & \text{(Deletion from A)} \\ \text{DP}[i, j-1] - \text{gap\_penalty} & \text{(Insertion into B)} \end{cases}$$
3. **Split & Merge Post-Processing:**
   - Handles legal re-drafting where:
     - One long clause in A is split into multiple sub-clauses in B (**1-many split**).
     - Multiple sub-clauses in A are merged into a single clause in B (**many-1 merge**).
   - Evaluates whether grouping consecutive gaps produces a higher average similarity than leaving them as orphan edits.

---

### Step 4: Rule-Augmented Severity Classification
**Source Files:** `src/semdiff/classifier/rule_checker.py`, `src/semdiff/classifier/pipeline.py`, `src/semdiff/classifier/distilbert_classifier.py`

Each aligned unit is classified into the legal severity taxonomy:

| Tier | Legal Meaning | Examples |
|:---|:---|:---|
| **`cosmetic`** | Pure paraphrase / translation with **no change** in legal obligation | `"shall submit"` $\to$ `"will submit"`; English $\to$ Tamil translation |
| **`substantive`** | Modifies numbers, thresholds, dates, or deadlines under the **same obligation direction** | `Rs. 500 cr` $\to$ `Rs. 750 cr`; `30 days` $\to$ `45 days` |
| **`critical`** | Inverts obligation, alters prohibition, or adds/deletes duties | `"may"` $\to$ `"shall"`; `"without approval"` $\to$ `"with approval"`; Deletion / Insertion |

#### The Hybrid Classification Architecture:
1. **Rule-Based Feature Extraction (Auxiliary Signal):**
   - High-precision regex pattern matchers scan for explicit regulatory shifts:
     - **Monetary & Numeric:** `amount_changed`, `number_changed`
     - **Deadlines & Dates:** `date_changed`, `deadline_changed`
     - **Modal Verbs:** `modal_strengthened` (`may` $\to$ `shall`), `modal_weakened` (`shall` $\to$ `may`)
     - **Negations:** `negation_added` (`shall` $\to$ `shall not`), `negation_removed` (`no person shall` $\to$ `every person shall`)
2. **DistilBERT Multilingual Classifier:**
   - Model: `distilbert-base-multilingual-cased`
   - Input structure: `[CLS] sentence_a [SEP] sentence_b`
   - Uses inverse-frequency class weighting to protect against class imbalance.
3. **Conservative Merging Policy:**
   - Final Label = $\max(\text{Rule Hint}, \text{Model Label})$
   - If a learned model predicts `cosmetic` due to high token overlap, but the rule engine flags a hard negation flip (`negation_added`), the system **enforces `critical` severity**.

---

### Step 5: Multi-Version Amendment Chain Tracking
**Source File:** `src/semdiff/alignment/chain.py`

1. When given a chain of three or more documents ($v_1 \to v_2 \to v_3$):
   - Computes pairwise diffs: $\text{Diff}(v_1, v_2)$ and $\text{Diff}(v_2, v_3)$.
2. Follows each `clause_id` across transitions:
   - Example: A capital adequacy clause in $v_1$ is increased in $v_2$ (`substantive`), then has its reporting deadline extended in $v_3$ (`substantive`).
3. Generates a **`ClauseTimeline`** mapping each transition hop, its text before/after, similarity score, and final clause lifecycle state (`active`, `deleted`, `merged`, `split`).

---

### Step 6: Interactive HTML Diff & JSON Serialization
**Source Files:** `src/semdiff/reporting/html_report.py`, `src/semdiff/reporting/json_export.py`

1. **Interactive HTML Diff Report:**
   - Single standalone `.html` file with zero external dependencies.
   - Side-by-side aligned layout with color coding (Green for unchanged, Yellow for cosmetic, Orange for substantive, Red for critical).
   - Real-time client-side search and severity filter buttons (`All`, `Critical`, `Substantive`, `Cosmetic`).
   - Visual confidence bars and triggered regulatory rule badges (`🚩 amount_changed`, `🚩 negation_added`).
   - Integrated timeline tree visualizing multi-version amendment progression.
2. **JSON Export:**
   - Programmatic JSON containing all sentence tokens, indices, similarity scores, classifications, and lineage metadata for downstream integration.

---

## 4. End-to-End Walkthrough of a Real Regulatory Clause

```
[Input v1]: "Banks may extend credit facilities to micro and small enterprises up to Rs. 10 crore without prior approval."
[Input v2]: "Banks shall not extend credit facilities to micro and small enterprises up to Rs. 10 crore without prior approval."
                                      │
                                      ▼
[Step 1 - Segmentation]: 
  • Extracted Clause ID: "4(2)(a)"
  • Detected Script: "en"
  • Shielded Tokens: "Rs. 10 crore"
                                      │
                                      ▼
[Step 2 - Embedding]:
  • Vector v1: [-0.018, 0.042, ... ] (768-dim L2 normalized)
  • Vector v2: [-0.015, 0.039, ... ] (768-dim L2 normalized)
                                      │
                                      ▼
[Step 3 - Alignment]:
  • Needleman-Wunsch pairs them as 1-1 match (cosine similarity = 0.887)
                                      │
                                      ▼
[Step 4 - Classification]:
  • Rule signals triggered: 
      - modal_strengthened ("may" -> "shall")
      - negation_added ("shall" -> "shall not")
      - obligation_changed
  • Final Severity: CRITICAL (Confidence: 0.98)
                                      │
                                      ▼
[Step 5 & 6 - Output]:
  • Rendered in red in reports/diff.html
  • Badges: [CRITICAL] [1-1] [🚩 negation_added] [🚩 modal_strengthened]
  • Appended to Section 4(2)(a)'s multi-version amendment timeline
```

---

## 5. Summary of Available Interfaces

| Interface | Command | Output |
|:---|:---|:---|
| **Terminal & HTML Demo** | `python demo.py` | Full terminal table + visual `reports/full_demo.html` |
| **CLI Tool** | `python -m semdiff.cli compare a.txt b.txt` | Console summary table + `reports/diff.html` |
| **Multi-Version CLI** | `python -m semdiff.cli compare v1.txt v2.txt v3.txt` | Paired HTML diffs across the entire chain |
| **Web UI (Streamlit)** | `streamlit run src/semdiff/ui.py` | Interactive web dashboard at `http://localhost:8501` |
| **REST API (FastAPI)** | `uvicorn src.semdiff.api:app --reload` | OpenAPI REST server at `http://localhost:8000` |
| **Evaluation Suite** | `python -X utf8 evaluation/eval_baselines.py --demo` | Baseline benchmarks & metrics report |

