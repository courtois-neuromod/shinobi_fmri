"""
Logging utilities for Shinobi fMRI analysis pipeline.

Provides structured logging with both console output and detailed log files
for debugging and tracking analysis computations.
"""

import os
import os.path as op
import logging
import sys
from datetime import datetime
from typing import Optional, List, Dict

class ProcessingSummary:
    """Track processing statistics for summary report."""

    def __init__(self):
        self.computed = []
        self.skipped = []
        self.errors = []
        self.warnings = []

    def add_computed(self, item: str):
        self.computed.append(item)

    def add_skipped(self, item: str):
        self.skipped.append(item)

    def add_error(self, item: str, error: str):
        self.errors.append((item, error))

    def add_warning(self, item: str, warning: str):
        self.warnings.append((item, warning))

    def get_summary_text(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("PROCESSING SUMMARY")
        lines.append("="*80)
        lines.append(f"✓ Computed:  {len(self.computed)}")
        lines.append(f"⊘ Skipped:   {len(self.skipped)}")
        lines.append(f"⚠ Warnings:  {len(self.warnings)}")
        lines.append(f"✗ Errors:    {len(self.errors)}")

        if self.computed:
            lines.append(f"\nComputed ({len(self.computed)}):")
            for item in self.computed[:10]:  # Show first 10
                lines.append(f"  ✓ {item}")
            if len(self.computed) > 10:
                lines.append(f"  ... and {len(self.computed) - 10} more")

        if self.skipped:
            lines.append(f"\nSkipped ({len(self.skipped)}):")
            for item in self.skipped[:5]:
                lines.append(f"  ⊘ {item}")
            if len(self.skipped) > 5:
                lines.append(f"  ... and {len(self.skipped) - 5} more")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for item, warning in self.warnings:
                lines.append(f"  ⚠ {item}: {warning}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for item, error in self.errors:
                lines.append(f"  ✗ {item}: {error}")

        lines.append("="*80 + "\n")
        return "\n".join(lines)


class ShinobiLogger:
    """Logger for Shinobi analysis with console and file output."""

    def __init__(self, log_name: str, subject: str = None, session: Optional[str] = None,
                 condition: Optional[str] = None, log_dir: Optional[str] = None,
                 verbosity: int = logging.INFO):
        """
        Initialize Shinobi logger.

        Args:
            log_name: Name of the logger/task (e.g. 'GLM_session', 'MVPA')
            subject: Subject ID (e.g., 'sub-01')
            session: Session ID (e.g., 'ses-001'), optional
            condition: Condition name (e.g., 'HIT'), optional
            log_dir: Directory for log files. If None, uses './logs/<log_name>/'
            verbosity: Logging level for console output (default: logging.INFO)
        """
        self.subject = subject
        self.session = session
        self.condition = condition
        self.log_name = log_name
        self.summary = ProcessingSummary()

        # Create log directory
        if log_dir is None:
            # Clean log_name for directory usage if needed, but assuming reasonable input
            log_dir = op.join(".", "logs", log_name)
        os.makedirs(log_dir, exist_ok=True)

        # Create log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_parts = []
        if subject: log_parts.append(subject)
        if session: log_parts.append(session)
        if condition: log_parts.append(condition)
        log_parts.append(timestamp)
        
        # If no identifiable parts, just use timestamp
        if not log_parts:
            log_parts = [timestamp]

        log_filename = op.join(log_dir, f"{'_'.join(log_parts)}.log")
        self.log_filename = log_filename

        # Create logger
        self.logger = logging.getLogger(f"{log_name}_{'_'.join(log_parts)}")
        self.logger.setLevel(logging.DEBUG) # Catch all, filter by handlers

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # File handler - detailed logging (DEBUG level always)
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler - configurable verbosity
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(verbosity)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Log initialization
        self.info(f"="*60)
        self.info(f"Logging initialized for {log_name}")
        if subject: self.info(f"Subject: {subject}")
        if session: self.info(f"Session: {session}")
        if condition: self.info(f"Condition: {condition}")
        self.info(f"Log file: {log_filename}")
        self.info(f"="*60)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str):
        """Log critical message."""
        self.logger.critical(msg)

    def log_file_check(self, filepath: str, file_type: str):
        """
        Log file existence check.
        """
        exists = op.exists(filepath)
        if exists:
            self.debug(f"✓ {file_type} file found: {filepath}")
        else:
            self.error(f"✗ {file_type} file NOT FOUND: {filepath}")
        return exists

    def log_computation_start(self, computation_type: str, output_file: str):
        """Log start of computation."""
        self.info(f"\n{'='*60}")
        self.info(f"Starting: {computation_type}")
        self.info(f"Output: {output_file}")
        self.info(f"{'='*60}")

    def log_computation_success(self, computation_type: str, output_file: str):
        """Log successful completion of computation."""
        self.info(f"✓ SUCCESS: {computation_type}")
        self.info(f"  Saved to: {output_file}")
        if op.exists(output_file):
            file_size = os.path.getsize(output_file)
            self.debug(f"  File size: {file_size:,} bytes")
        self.summary.add_computed(computation_type)

    def log_computation_skip(self, computation_type: str, output_file: str):
        """Log skipping of computation (file already exists)."""
        self.info(f"⊘ SKIP: {computation_type} (already exists)")
        self.debug(f"  File: {output_file}")
        self.summary.add_skipped(computation_type)

    def log_computation_error(self, computation_type: str, error: Exception):
        """Log computation error."""
        self.error(f"✗ ERROR: {computation_type}")
        error_msg = f"{type(error).__name__}: {str(error)}"
        self.error(f"  Exception: {error_msg}")

        # Log full traceback to file only
        import traceback
        self.debug("Full traceback:")
        self.debug(traceback.format_exc())

        self.summary.add_error(computation_type, error_msg)

    def print_summary(self):
        """Print and log the processing summary."""
        summary_text = self.summary.get_summary_text()

        # Print to console (always, unless extremely silent?)
        # Let's use logger.info so it respects verbosity, but maybe force print?
        # User requested summary at the end.
        print(summary_text)

        # Log to file
        for line in summary_text.split('\n'):
            if line.strip():
                self.logger.info(line)

        if self.summary.errors:
            self.warning(f"\nDetailed error logs saved to: {self.log_filename}")

    def close(self):
        """Print summary and close all handlers."""
        self.print_summary()

        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
