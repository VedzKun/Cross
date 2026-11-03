"""
JSON export for DiffResult and amendment chain timelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Any

from ..models import DiffResult, ClauseTimeline


def _to_dict(obj: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def export_diff_to_json(
    diff: DiffResult,
    timelines: Optional[List[ClauseTimeline]] = None,
    output_path: Optional[Path] = None,
    indent: int = 2,
) -> str:
    """
    Export a DiffResult (and optionally amendment timelines) to JSON.
    Returns the JSON string and optionally writes it to output_path.
    """
    payload = {
        "diff": _to_dict(diff),
        "timelines": _to_dict(timelines) if timelines else [],
    }
    serialized = json.dumps(payload, indent=indent, ensure_ascii=False, default=str)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(serialized, encoding="utf-8")
    return serialized

