"""
Reporting module: HTML side-by-side diff report and JSON export.
"""

from .html_report import HtmlReportGenerator, generate_html_report
from .json_export import export_diff_to_json

__all__ = ["HtmlReportGenerator", "generate_html_report", "export_diff_to_json"]

