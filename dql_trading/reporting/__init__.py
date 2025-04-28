"""
Reporting module for the DQL Trading Framework.

This module provides functionality for generating PDF reports of trading strategy
performance and results.
"""

# Re-export key symbols for convenient import paths
# We now rely on the fully-featured implementation in reporting.py which
# supports dynamic TOC handling.

from .reporting import TradingReport, TradingReportPDF  # type: ignore F401
from .loaders import (
    load_metrics,
    find_visualization_files,
    load_baseline_comparison,
)  # noqa: F401 