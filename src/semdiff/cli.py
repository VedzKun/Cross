"""
semdiff CLI — Compare regulatory documents semantically.

Usage:
    semdiff compare a.pdf b.pdf
    semdiff compare a.pdf b.pdf c.pdf          # multi-version chain
    semdiff compare a.pdf b.pdf --output report.html --json result.json
    semdiff compare a.pdf b.pdf --mode rule_only
"""

from __future__ import annotations

import logging
import sys
import io
import time
from pathlib import Path
from typing import List, Optional

# Reconfigure Windows console to UTF-8
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="semdiff",
    help="Cross-Lingual Semantic Diff for Regulatory Documents",
    add_completion=False,
)
console = Console(legacy_windows=False)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")


@app.command("compare")
def compare(
    documents: List[Path] = typer.Argument(
        ..., help="Two or more document files (PDF, DOCX, TXT) in version order"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path for HTML report (default: reports/diff_<timestamp>.html)"
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json", "-j", help="Path for JSON export"
    ),
    mode: str = typer.Option(
        "rule_only", "--mode", "-m",
        help="Classification mode: rule_only | model_only | combined"
    ),
    classifier_checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", "-c",
        help="Path to fine-tuned DistilBERT classifier checkpoint"
    ),
    sim_threshold: float = typer.Option(0.50, "--sim-threshold", help="Similarity threshold for alignment"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable embedding disk cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    open_report: bool = typer.Option(False, "--open", help="Open HTML report in browser after generation"),
):
    """
    Compare two or more regulatory document versions and produce a semantic diff report.

    Examples:

        semdiff compare v1.pdf v2.pdf

        semdiff compare v1.pdf v2.pdf v3.pdf --output chain_report.html

        semdiff compare rbi_v1.txt rbi_v2.txt --mode combined --json diff.json
    """
    _setup_logging(verbose)

    if len(documents) < 2:
        console.print("[red]Error:[/red] At least two documents are required.")
        raise typer.Exit(1)

    for doc_path in documents:
        if not doc_path.exists():
            console.print(f"[red]Error:[/red] File not found: {doc_path}")
            raise typer.Exit(1)

    start_time = time.time()

    # Lazy imports to keep startup fast
    from .ingestion.readers import DocumentReader
    from .embedding.labse import LaBSEEmbedder
    from .alignment.aligner import SemanticAligner
    from .alignment.chain import AmendmentChain
    from .classifier.pipeline import ClassificationPipeline
    from .reporting.html_report import generate_html_report
    from .reporting.json_export import export_diff_to_json
    from .config import default_settings

    default_settings.ensure_directories()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # Step 1: Ingest documents
        task = progress.add_task("[Ingest] Ingesting documents...", total=None)
        reader = DocumentReader()
        docs = []
        for i, doc_path in enumerate(documents):
            version = f"v{i+1}"
            doc = reader.read(doc_path, version=version)
            docs.append(doc)
            progress.console.print(
                f"  [green][OK][/green] {doc_path.name}: {len(doc.sentences)} sentences ({version})"
            )
        progress.remove_task(task)

        # Step 2: Initialize embedder
        task = progress.add_task("[Embed] Loading embedder...", total=None)
        embedder = LaBSEEmbedder(
            enable_cache=not no_cache,
            cache_db_path=default_settings.embedding_cache_db,
        )
        progress.remove_task(task)

        # Step 3: Run alignment chain
        task = progress.add_task("[Align] Running semantic alignment...", total=None)
        aligner = SemanticAligner(sim_threshold=sim_threshold)
        chain = AmendmentChain(embedder=embedder, aligner=aligner)
        for doc in docs:
            chain.add_version(doc)
        diffs = chain.get_diffs()
        timelines = chain.get_timelines()
        progress.remove_task(task)

        # Step 4: Classify
        task = progress.add_task(f"[Classify] Classifying ({mode})...", total=None)
        classifier = ClassificationPipeline(
            mode=mode,
            classifier_checkpoint=classifier_checkpoint,
        )
        for diff in diffs:
            classifier.classify_diff_result(diff)
        progress.remove_task(task)

    # Determine output paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output is None:
        output = default_settings.reports_dir / f"diff_{timestamp}.html"
    if json_out is None and len(sys.argv) > 1:
        json_out = None  # only write JSON if explicitly requested

    elapsed = time.time() - start_time

    # Generate output for each consecutive pair
    for i, (diff, doc_a, doc_b) in enumerate(zip(diffs, docs[:-1], docs[1:])):
        pair_output = output
        if len(diffs) > 1:
            pair_output = output.parent / f"{output.stem}_v{i+1}v{i+2}{output.suffix}"

        generate_html_report(
            diff,
            doc_a_label=f"{doc_a.filename} ({doc_a.version})",
            doc_b_label=f"{doc_b.filename} ({doc_b.version})",
            timelines=timelines if i == len(diffs) - 1 else None,
            output_path=pair_output,
        )
        console.print(f"[green][OK][/green] HTML report: [link={pair_output}]{pair_output}[/link]")

        if json_out:
            pair_json = json_out
            if len(diffs) > 1:
                pair_json = json_out.parent / f"{json_out.stem}_v{i+1}v{i+2}{json_out.suffix}"
            export_diff_to_json(diff, timelines if i == len(diffs) - 1 else None, pair_json)
            console.print(f"[green][OK][/green] JSON export: {pair_json}")

    # Print summary table
    _print_summary(diffs, elapsed, console)

    if open_report:
        import webbrowser
        webbrowser.open(str(output))


def _print_summary(diffs, elapsed, con):
    table = Table(title="Summary", show_header=True, header_style="bold #1e3a5f")
    table.add_column("Pair")
    table.add_column("Unchanged", style="green")
    table.add_column("Cosmetic", style="yellow")
    table.add_column("Substantive", style="orange3")
    table.add_column("Critical", style="red")
    table.add_column("Insertions")
    table.add_column("Deletions")

    for i, diff in enumerate(diffs):
        sev = diff.stats.get("severity_counts", {})
        atype = diff.stats.get("alignment_type_counts", {})
        table.add_row(
            f"v{i+1}→v{i+2}",
            str(sev.get("unchanged", 0)),
            str(sev.get("cosmetic", 0)),
            str(sev.get("substantive", 0)),
            str(sev.get("critical", 0)),
            str(atype.get("insert", 0)),
            str(atype.get("delete", 0)),
        )

    con.print(table)
    con.print(f"\n[dim]Total time: {elapsed:.1f}s[/dim]")


@app.command("train")
def train_classifier(
    train_csv: Path = typer.Argument(..., help="Path to training CSV (pairs with label column)"),
    val_csv: Optional[Path] = typer.Option(None, help="Validation CSV"),
    output_dir: Path = typer.Option(Path("models/classifier"), help="Where to save the fine-tuned model"),
    epochs: int = typer.Option(5, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
    label_col: str = typer.Option("label", help="CSV column containing the label"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Fine-tune the DistilBERT severity classifier on labeled pairs."""
    _setup_logging(verbose)

    import csv
    from .classifier.distilbert_classifier import SeverityClassifier, LABEL2ID

    def load_csv(path: Path):
        rows = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get(label_col, "").strip()
                if lbl in LABEL2ID:
                    rows.append(row)
        return rows

    console.print(f"Loading training data from [cyan]{train_csv}[/cyan]")
    train_rows = load_csv(train_csv)
    console.print(f"  {len(train_rows)} training pairs")

    val_rows = []
    if val_csv:
        val_rows = load_csv(val_csv)
        console.print(f"  {len(val_rows)} validation pairs")

    ta = [r["text_a"] for r in train_rows]
    tb = [r["text_b"] for r in train_rows]
    tl = [LABEL2ID[r[label_col]] for r in train_rows]

    weights = SeverityClassifier.compute_class_weights(tl)
    console.print(f"Class weights: {dict(zip(['cosmetic','substantive','critical'], [round(w,2) for w in weights]))}")

    clf = SeverityClassifier(checkpoint_dir=output_dir)
    clf.train(
        ta, tb, tl,
        num_epochs=epochs,
        batch_size=batch_size,
        class_weights=weights,
        val_texts_a=[r["text_a"] for r in val_rows] if val_rows else None,
        val_texts_b=[r["text_b"] for r in val_rows] if val_rows else None,
        val_labels=[LABEL2ID[r[label_col]] for r in val_rows] if val_rows else None,
    )
    clf.save(output_dir)
    console.print(f"[green]✓[/green] Classifier saved to {output_dir}")


@app.command("dataset")
def build_dataset(
    cosmetic: int = typer.Option(150),
    substantive: int = typer.Option(100),
    critical: int = typer.Option(100),
    output_dir: Path = typer.Option(Path("data/synthetic")),
    split: bool = typer.Option(True, help="Also run train/val/test split"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate synthetic training dataset and optionally split it."""
    _setup_logging(verbose)
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "data/scripts/build_dataset.py",
            "--cosmetic", str(cosmetic),
            "--substantive", str(substantive),
            "--critical", str(critical),
            "--output-dir", str(output_dir),
        ],
        capture_output=True, text=True
    )
    console.print(result.stdout)
    if result.returncode != 0:
        console.print(f"[red]{result.stderr}[/red]")
        raise typer.Exit(1)

    if split:
        result2 = subprocess.run(
            [
                sys.executable, "data/scripts/split_dataset.py",
                "--input", str(output_dir / "pairs.csv"),
            ],
            capture_output=True, text=True
        )
        console.print(result2.stdout)
        if result2.returncode != 0:
            console.print(f"[red]{result2.stderr}[/red]")


def main():
    app()


if __name__ == "__main__":
    main()

