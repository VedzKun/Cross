"""
Efficiency benchmark: latency and memory usage on CPU.

Measures:
  - Document ingestion time vs. document size
  - Embedding time (LaBSE) for varying batch sizes
  - DP alignment time for varying document lengths
  - End-to-end pipeline time
  - Peak RAM usage via tracemalloc

Usage:
    python evaluation/efficiency.py
    python evaluation/efficiency.py --n-sents 50 --skip-labse   # quick run without model
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def format_mem(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 ** 2:
        return f"{bytes_ / 1024:.1f} KB"
    elif bytes_ < 1024 ** 3:
        return f"{bytes_ / 1024**2:.1f} MB"
    return f"{bytes_ / 1024**3:.2f} GB"


def time_it(fn, *args, **kwargs) -> Tuple[float, any]:
    """Run fn and return (elapsed_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def generate_synthetic_text(n_sentences: int, language: str = "en") -> str:
    """Generate a regulatory document with n_sentences."""
    if language == "en":
        templates = [
            "Section {i}: Every registered entity shall comply with the provisions of sub-regulation {i}(1) within 30 days of the notification date.",
            "Clause {i}: No person shall engage in transactions exceeding Rs. {amount} crore without prior written approval from the designated authority.",
            "Rule {i}({sub}): Banks and financial institutions must maintain a minimum capital adequacy ratio of {ratio} per cent on a continuous basis.",
            "Article {i}: The compliance report shall be submitted quarterly to the Reserve Bank within {days} working days of the quarter end.",
            "Regulation {i}: Any breach of these guidelines shall attract a penalty not exceeding Rs. {fine} lakh per day of continuing default.",
        ]
    elif language == "hi":
        templates = [
            "धारा {i}: प्रत्येक पंजीकृत संस्था उप-विनियम {i}(1) के प्रावधानों का अधिसूचना की तारीख से 30 दिनों के भीतर अनुपालन करेगी।",
            "खंड {i}: कोई भी व्यक्ति नामित प्राधिकरण की पूर्व लिखित अनुमति के बिना रु. {amount} करोड़ से अधिक के लेनदेन में शामिल नहीं होगा।",
        ]
    else:
        templates = [
            "பிரிவு {i}: ஒவ்வொரு பதிவு செய்யப்பட்ட நிறுவனமும் அறிவிப்பு தேதியிலிருந்து 30 நாட்களுக்குள் துணை-விதிமுறை {i}(1) விதிகளை பின்பற்ற வேண்டும்.",
        ]

    lines = []
    for i in range(1, n_sentences + 1):
        template = templates[i % len(templates)]
        line = template.format(
            i=i,
            sub=chr(ord('a') + (i % 26)),
            amount=100 + i * 10,
            ratio=10 + (i % 5),
            days=15 + (i % 15),
            fine=1 + (i % 10),
        )
        lines.append(line)
    return "\n\n".join(lines)


def benchmark_ingestion(n_sentences_list: List[int]) -> None:
    print("\n=== Ingestion Benchmark ===")
    print(f"{'N Sentences':>12} {'Time (s)':>12} {'Peak RAM':>12} {'Sents/s':>12}")
    print("-" * 52)

    from semdiff.ingestion.splitter import split_legal_text

    for n in n_sentences_list:
        text = generate_synthetic_text(n)

        tracemalloc.start()
        elapsed, sents = time_it(split_legal_text, text, "bench_doc")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rate = len(sents) / elapsed if elapsed > 0 else 0
        print(f"{n:>12} {elapsed:>12.3f} {format_mem(peak):>12} {rate:>12.0f}")


def benchmark_alignment(n_list: List[int]) -> None:
    print("\n=== DP Alignment Benchmark ===")
    print(f"{'N (each doc)':>14} {'Time (s)':>12} {'Peak RAM':>12} {'Pairs/s':>12}")
    print("-" * 54)

    from semdiff.alignment.aligner import SemanticAligner
    from semdiff.ingestion.splitter import split_legal_text

    for n in n_list:
        text_a = generate_synthetic_text(n)
        text_b = generate_synthetic_text(n)
        sents_a = split_legal_text(text_a, "a")
        sents_b = split_legal_text(text_b, "b")

        # Mock similarity matrix (random)
        np.random.seed(0)
        na, nb = len(sents_a), len(sents_b)
        sim = np.random.uniform(0.3, 0.95, (na, nb)).astype(np.float32)

        aligner = SemanticAligner()
        tracemalloc.start()
        elapsed, diff = time_it(aligner.align, sents_a, sents_b, sim, "a", "b")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        pairs_per_sec = len(diff.aligned_pairs) / elapsed if elapsed > 0 else 0
        print(f"{n:>14} {elapsed:>12.3f} {format_mem(peak):>12} {pairs_per_sec:>12.0f}")


def benchmark_rule_checker(n_list: List[int]) -> None:
    print("\n=== Rule Checker Benchmark ===")
    print(f"{'N Pairs':>10} {'Time (s)':>12} {'Pairs/s':>12}")
    print("-" * 36)

    from semdiff.classifier.rule_checker import RuleBasedChecker
    from semdiff.ingestion.splitter import split_legal_text

    checker = RuleBasedChecker()

    for n in n_list:
        texts_a = [
            generate_synthetic_text(1).split("\n\n")[0] for _ in range(n)
        ]
        texts_b = [
            generate_synthetic_text(1).split("\n\n")[0] for _ in range(n)
        ]

        elapsed, results = time_it(
            lambda: [checker.compare(a, b) for a, b in zip(texts_a, texts_b)]
        )
        rate = n / elapsed if elapsed > 0 else 0
        print(f"{n:>10} {elapsed:>12.3f} {rate:>12.0f}")


def benchmark_embedding(n_list: List[int], skip_labse: bool = False) -> None:
    if skip_labse:
        print("\n=== Embedding Benchmark (skipped) ===")
        return

    print("\n=== LaBSE Embedding Benchmark (CPU) ===")
    print(f"{'N Sentences':>12} {'Time (s)':>12} {'Peak RAM':>12} {'Sents/s':>12}")
    print("-" * 52)

    try:
        from semdiff.embedding.labse import LaBSEEmbedder
        embedder = LaBSEEmbedder(enable_cache=False)

        for n in n_list:
            texts = [generate_synthetic_text(1).split("\n\n")[0] for _ in range(n)]

            tracemalloc.start()
            elapsed, embs = time_it(embedder.embed_sentences, texts)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            rate = n / elapsed if elapsed > 0 else 0
            print(f"{n:>12} {elapsed:>12.3f} {format_mem(peak):>12} {rate:>12.1f}")

    except Exception as e:
        print(f"  (LaBSE unavailable: {e})")


def main():
    parser = argparse.ArgumentParser(description="CPU efficiency benchmark.")
    parser.add_argument("--n-sents", type=int, default=50, help="Max sentences for benchmark")
    parser.add_argument("--skip-labse", action="store_true", help="Skip LaBSE embedding benchmark")
    args = parser.parse_args()

    n_max = args.n_sents
    ingestion_sizes = [10, 25, 50, min(100, n_max), min(200, n_max)]
    ingestion_sizes = sorted(set(s for s in ingestion_sizes if s <= n_max))
    alignment_sizes = [10, 25, 50, min(100, n_max)]
    alignment_sizes = sorted(set(s for s in alignment_sizes if s <= n_max))
    rule_sizes = [50, 100, 250, 500, min(1000, n_max * 20)]
    embedding_sizes = [5, 10, 20, min(50, n_max)]

    print("=" * 60)
    print("EFFICIENCY BENCHMARK (CPU)")
    print("=" * 60)

    benchmark_ingestion(ingestion_sizes)
    benchmark_alignment(alignment_sizes)
    benchmark_rule_checker(rule_sizes)
    benchmark_embedding(embedding_sizes, skip_labse=args.skip_labse)

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("Expected targets for a 4-core laptop CPU:")
    print("  Ingestion:     > 500 sentences/s")
    print("  DP alignment:  > 200 pairs/s for 50-sentence docs")
    print("  Rule checker:  > 5,000 pairs/s")
    print("  LaBSE embed:   > 20 sentences/s (no cache)")
    print("=" * 60)


if __name__ == "__main__":
    main()

