"""
HTML report generator: produces a self-contained, styled diff report
with side-by-side aligned view, color-coded severity, per-change confidence,
and amendment chain timeline.
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..models import (
    DiffResult, AlignedPair, AlignmentType, SeverityLevel,
    ClauseTimeline, ClauseLineageHop,
)

# Color palette per severity
SEVERITY_COLORS = {
    SeverityLevel.UNCHANGED: "#f0f9ff",
    SeverityLevel.COSMETIC: "#fefce8",
    SeverityLevel.SUBSTANTIVE: "#fff7ed",
    SeverityLevel.CRITICAL: "#fef2f2",
}
SEVERITY_BORDER = {
    SeverityLevel.UNCHANGED: "#bae6fd",
    SeverityLevel.COSMETIC: "#fde68a",
    SeverityLevel.SUBSTANTIVE: "#fdba74",
    SeverityLevel.CRITICAL: "#fca5a5",
}
SEVERITY_BADGE = {
    SeverityLevel.UNCHANGED: ("#0369a1", "#e0f2fe"),
    SeverityLevel.COSMETIC: ("#854d0e", "#fef9c3"),
    SeverityLevel.SUBSTANTIVE: ("#9a3412", "#ffedd5"),
    SeverityLevel.CRITICAL: ("#991b1b", "#fee2e2"),
}
ALIGNMENT_ICON = {
    AlignmentType.ONE_TO_ONE: "⇔",
    AlignmentType.ONE_TO_MANY: "⇉",
    AlignmentType.MANY_TO_ONE: "⇇",
    AlignmentType.INSERTION: "✚",
    AlignmentType.DELETION: "✖",
}
ALIGNMENT_LABEL = {
    AlignmentType.ONE_TO_ONE: "1-1",
    AlignmentType.ONE_TO_MANY: "split",
    AlignmentType.MANY_TO_ONE: "merge",
    AlignmentType.INSERTION: "insert",
    AlignmentType.DELETION: "delete",
}

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 14px; color: #1e293b; background: #f8fafc; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px; }
header { background: #1e3a5f; color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 24px; }
header h1 { font-size: 22px; font-weight: 700; }
header p { color: #94a3b8; font-size: 13px; margin-top: 4px; }
.stats-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.stat-card { background: white; border-radius: 10px; padding: 14px 18px; border: 1px solid #e2e8f0; min-width: 130px; }
.stat-card .val { font-size: 24px; font-weight: 700; }
.stat-card .lbl { font-size: 12px; color: #64748b; margin-top: 2px; }
.section-title { font-size: 17px; font-weight: 600; color: #334155; margin: 28px 0 12px 0; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0; }
table.diff-table { width: 100%; border-collapse: collapse; }
table.diff-table th { background: #334155; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }
table.diff-table td { vertical-align: top; padding: 10px 14px; }
.pair-row { border-radius: 8px; margin-bottom: 2px; }
.pair-row td { border-top: 2px solid transparent; border-bottom: 2px solid transparent; }
.cell-a { border-right: 1px solid #e2e8f0; }
.text-content { white-space: pre-wrap; line-height: 1.6; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-left: 4px; }
.align-badge { background: #e2e8f0; color: #334155; font-size: 11px; padding: 2px 6px; border-radius: 4px; }
.clause-tag { font-size: 11px; color: #64748b; margin-bottom: 4px; font-family: monospace; }
.conf-bar { height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: 6px; }
.conf-fill { height: 100%; border-radius: 2px; background: #3b82f6; }
.flags { margin-top: 6px; }
.flag-tag { display: inline-block; background: #fef3c7; color: #92400e; border: 1px solid #fde68a; font-size: 10px; padding: 1px 5px; border-radius: 4px; margin: 1px; }
.empty-cell { color: #94a3b8; font-style: italic; }
/* Timeline */
.timeline { list-style: none; padding: 0; }
.timeline-item { display: flex; gap: 14px; margin-bottom: 16px; }
.timeline-line { display: flex; flex-direction: column; align-items: center; }
.timeline-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
.timeline-connector { width: 2px; flex: 1; background: #e2e8f0; margin-top: 2px; }
.timeline-content { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; flex: 1; }
.timeline-header { display: flex; justify-content: space-between; font-size: 12px; color: #64748b; margin-bottom: 6px; }
.timeline-texts { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px; }
.timeline-text-box { background: #f8fafc; border-radius: 4px; padding: 6px 8px; color: #475569; }
.filter-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }
.filter-btn { padding: 4px 12px; border-radius: 20px; border: 1px solid #cbd5e1; background: white; cursor: pointer; font-size: 12px; }
.filter-btn.active { background: #1e3a5f; color: white; border-color: #1e3a5f; }
.search-box { padding: 4px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 12px; width: 200px; }
@media print { .filter-bar { display: none; } }
"""

JS = """
function filterRows(severity) {
  const btns = document.querySelectorAll('.filter-btn');
  btns.forEach(b => b.classList.remove('active'));
  const clicked = document.querySelector(`.filter-btn[data-sev="${severity}"]`);
  if (clicked) clicked.classList.add('active');
  const rows = document.querySelectorAll('tr.pair-row');
  rows.forEach(row => {
    if (severity === 'all' || row.dataset.severity === severity) {
      row.style.display = '';
    } else {
      row.style.display = 'none';
    }
  });
}
function searchPairs(query) {
  const q = query.toLowerCase();
  const rows = document.querySelectorAll('tr.pair-row');
  rows.forEach(row => {
    const text = row.textContent.toLowerCase();
    row.style.display = (!q || text.includes(q)) ? '' : 'none';
  });
}
"""


class HtmlReportGenerator:
    """Generates a self-contained HTML report for a DiffResult."""

    def generate(
        self,
        diff: DiffResult,
        doc_a_label: str = "Version A",
        doc_b_label: str = "Version B",
        timelines: Optional[List[ClauseTimeline]] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate HTML string. Optionally write to file."""
        body = self._render(diff, doc_a_label, doc_b_label, timelines)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(body, encoding="utf-8")
        return body

    def _render(
        self,
        diff: DiffResult,
        label_a: str,
        label_b: str,
        timelines: Optional[List[ClauseTimeline]],
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        stats = diff.stats
        sev_counts = stats.get("severity_counts", {})
        atype_counts = stats.get("alignment_type_counts", {})

        stat_cards_html = self._stat_cards(sev_counts, atype_counts)
        diff_table_html = self._diff_table(diff.aligned_pairs, label_a, label_b)
        timeline_html = self._timeline_section(timelines) if timelines else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SemDiff Report — {_h(label_a)} vs {_h(label_b)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<header>
  <h1>📄 Cross-Lingual Semantic Diff Report</h1>
  <p>{_h(label_a)} &nbsp;↔&nbsp; {_h(label_b)} &nbsp;|&nbsp; Generated: {now}</p>
</header>
{stat_cards_html}
<div class="section-title">Aligned Changes</div>
{diff_table_html}
{timeline_html}
</div>
<script>{JS}</script>
</body>
</html>"""

    def _stat_cards(self, sev_counts: dict, atype_counts: dict) -> str:
        cards = [
            ("Total Pairs", sum(sev_counts.values()), "#1e3a5f"),
            ("🟡 Cosmetic", sev_counts.get("cosmetic", 0), "#854d0e"),
            ("🟠 Substantive", sev_counts.get("substantive", 0), "#9a3412"),
            ("🔴 Critical", sev_counts.get("critical", 0), "#991b1b"),
            ("✅ Unchanged", sev_counts.get("unchanged", 0), "#065f46"),
            ("➕ Insertions", atype_counts.get("insert", 0), "#1d4ed8"),
            ("✖ Deletions", atype_counts.get("delete", 0), "#7c3aed"),
        ]
        html_parts = ['<div class="stats-row">']
        for label, val, color in cards:
            html_parts.append(
                f'<div class="stat-card"><div class="val" style="color:{color}">{val}</div>'
                f'<div class="lbl">{_h(label)}</div></div>'
            )
        html_parts.append("</div>")
        return "\n".join(html_parts)

    def _diff_table(self, pairs: List[AlignedPair], label_a: str, label_b: str) -> str:
        severity_opts = ["all", "critical", "substantive", "cosmetic", "unchanged"]
        filter_btns = " ".join(
            f'<button class="filter-btn{" active" if s=="all" else ""}" '
            f'data-sev="{s}" onclick="filterRows(\'{s}\')">{s.title()}</button>'
            for s in severity_opts
        )
        filter_bar = (
            f'<div class="filter-bar">'
            f'<span style="font-size:12px;color:#64748b">Filter:</span>{filter_btns}'
            f'<input class="search-box" placeholder="Search text..." oninput="searchPairs(this.value)"/>'
            f'</div>'
        )

        rows_html = []
        for pair in pairs:
            bg = SEVERITY_COLORS.get(pair.severity, "#ffffff")
            border = SEVERITY_BORDER.get(pair.severity, "#e2e8f0")
            sev_col, sev_bg = SEVERITY_BADGE.get(pair.severity, ("#334155", "#f1f5f9"))
            icon = ALIGNMENT_ICON.get(pair.alignment_type, "?")
            alabel = ALIGNMENT_LABEL.get(pair.alignment_type, "?")

            sev_badge = (
                f'<span class="badge" style="color:{sev_col};background:{sev_bg}">'
                f'{pair.severity.value}</span>'
            )
            align_badge = f'<span class="align-badge">{icon} {alabel}</span>'

            text_a = _h(pair.text_a) if pair.text_a else '<span class="empty-cell">[not present]</span>'
            text_b = _h(pair.text_b) if pair.text_b else '<span class="empty-cell">[not present]</span>'

            clause_a = ""
            if pair.sentences_a and pair.sentences_a[0].clause_id:
                clause_a = f'<div class="clause-tag">§ {_h(pair.sentences_a[0].clause_id)}</div>'
            clause_b = ""
            if pair.sentences_b and pair.sentences_b[0].clause_id:
                clause_b = f'<div class="clause-tag">§ {_h(pair.sentences_b[0].clause_id)}</div>'

            conf_pct = int(pair.confidence * 100)
            conf_html = (
                f'<div class="conf-bar" title="Confidence: {conf_pct}%">'
                f'<div class="conf-fill" style="width:{conf_pct}%"></div></div>'
            )

            flags_html = ""
            if pair.rule_flags:
                flags_html = (
                    '<div class="flags">'
                    + " ".join(f'<span class="flag-tag">{_h(f)}</span>' for f in pair.rule_flags)
                    + "</div>"
                )

            sim_str = f"{pair.similarity_score:.3f}"

            rows_html.append(
                f'<tr class="pair-row" data-severity="{pair.severity.value}" '
                f'style="background:{bg};border-left:4px solid {border};">'
                f'<td class="cell-a">{clause_a}<div class="text-content">{text_a}</div></td>'
                f'<td class="cell-a">{clause_b}<div class="text-content">{text_b}</div></td>'
                f'<td style="white-space:nowrap;">{sev_badge}{align_badge}<br/>'
                f'<span style="font-size:11px;color:#64748b">sim: {sim_str}</span>'
                f'{conf_html}{flags_html}</td>'
                f'</tr>'
            )

        return (
            filter_bar
            + '<table class="diff-table">'
            + f'<thead><tr><th style="width:44%">{_h(label_a)}</th>'
            + f'<th style="width:44%">{_h(label_b)}</th>'
            + '<th style="width:12%">Analysis</th></tr></thead>'
            + '<tbody>'
            + "\n".join(rows_html)
            + "</tbody></table>"
        )

    def _timeline_section(self, timelines: List[ClauseTimeline]) -> str:
        if not timelines:
            return ""

        items = []
        for tl in timelines[:50]:  # cap at 50 timelines
            if not tl.history:
                continue
            hops_html = self._render_hops(tl.history)
            status_color = {
                "active": "#16a34a",
                "deleted": "#dc2626",
                "merged": "#d97706",
                "split": "#7c3aed",
            }.get(tl.final_status, "#64748b")
            items.append(
                f'<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin-bottom:12px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:10px;">'
                f'<strong style="font-family:monospace">§ {_h(tl.clause_id)}</strong>'
                f'<span class="badge" style="background:{status_color}22;color:{status_color}">'
                f'{tl.final_status}</span></div>'
                f'{hops_html}</div>'
            )

        if not items:
            return ""

        return (
            '<div class="section-title">Amendment Chain Timeline</div>'
            + "\n".join(items)
        )

    def _render_hops(self, hops: List[ClauseLineageHop]) -> str:
        parts = []
        for hop in hops:
            dot_color = SEVERITY_BORDER.get(hop.severity, "#e2e8f0")
            parts.append(
                f'<div style="display:flex;gap:10px;margin-bottom:8px;">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{dot_color};flex-shrink:0;margin-top:4px;"></div>'
                f'<div style="flex:1">'
                f'<div style="font-size:11px;color:#64748b;margin-bottom:4px;">'
                f'{_h(hop.from_version)} → {_h(hop.to_version)} &nbsp;|&nbsp; '
                f'<span class="badge" style="background:#f1f5f9;color:#334155">{hop.alignment_type.value}</span>'
                f'&nbsp; sim: {hop.similarity:.3f}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">'
                f'<div style="background:#f8fafc;border-radius:4px;padding:6px;color:#475569">{_h(hop.text_before[:200])}</div>'
                f'<div style="background:#f8fafc;border-radius:4px;padding:6px;color:#475569">{_h(hop.text_after[:200])}</div>'
                f'</div></div></div>'
            )
        return "\n".join(parts)


def _h(text: str) -> str:
    """HTML-escape a string."""
    return _html.escape(str(text))


def generate_html_report(
    diff: DiffResult,
    doc_a_label: str = "Version A",
    doc_b_label: str = "Version B",
    timelines: Optional[List[ClauseTimeline]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Convenience wrapper."""
    return HtmlReportGenerator().generate(diff, doc_a_label, doc_b_label, timelines, output_path)

