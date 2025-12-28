"""
Testing and validation tools for shinobi_fmri pipeline outputs.

This module provides tools to validate the completeness and integrity
of analysis outputs from the shinobi_fmri pipeline.
"""

from tests.utils import ValidationResult
from tests.validate_outputs import PipelineValidator

__all__ = ['ValidationResult', 'PipelineValidator']
