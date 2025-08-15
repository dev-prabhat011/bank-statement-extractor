"""Extractor package: modular bank statement extraction components.

This package provides:
- StatementExtractor: Main extraction class
- analysis: Transaction analysis functions
- exporters: Excel and XML export functions
- utils: Utility functions for parsing and formatting
- parsers: Bank-specific parsing logic
"""

__all__ = [
    "StatementExtractor",
    "analysis",
    "exporters", 
    "utils",
    "parsers",
]

from .extractor import StatementExtractor


