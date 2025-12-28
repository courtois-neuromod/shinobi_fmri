"""
Comprehensive Output Validation for shinobi_fmri Pipeline

This script validates all expected outputs from the analysis pipeline,
checking for completeness and optionally file integrity.

Usage:
    python tests/validate_outputs.py
    python tests/validate_outputs.py --subject sub-01
    python tests/validate_outputs.py --analysis-type glm_session
    python tests/validate_outputs.py --check-integrity

    # Via invoke:
    invoke validate.outputs
    invoke validate.outputs --subject sub-01 --analysis-type mvpa
"""

import os
import os.path as op
import sys
import argparse
import json
import logging
from typing import Dict, List, Optional

# Add parent directory to path to import shinobi_fmri
sys.path.insert(0, op.dirname(op.dirname(op.abspath(__file__))))

from shinobi_fmri import config
from tests.utils import (
    ValidationResult,
    validate_nifti,
    validate_pickle,
    get_available_runs,
    get_subjects_from_raw_data,
    get_sessions_from_raw_data,
    load_correlation_matrix,
    validate_correlation_matrix,
    get_expected_beta_maps
)
from shinobi_fmri.utils.logger import AnalysisLogger


class PipelineValidator:
    """Validates all outputs from the shinobi_fmri analysis pipeline."""

    def __init__(
        self,
        data_path: str,
        subjects: Optional[List[str]] = None,
        analysis_types: Optional[List[str]] = None,
        check_integrity: bool = False,
        logger: Optional[AnalysisLogger] = None
    ):
        """
        Initialize validator.

        Args:
            data_path: Root data directory
            subjects: List of subjects to validate (None = all from config)
            analysis_types: List of analysis types to check (None = all)
                Options: ['glm_session', 'glm_subject', 'mvpa', 'correlations', 'figures']
            check_integrity: If True, validate file contents (slower)
            logger: Optional logger instance
        """
        self.data_path = data_path
        self.subjects = subjects or config.SUBJECTS
        self.conditions = config.CONDITIONS
        self.check_integrity = check_integrity
        self.logger = logger

        # Determine which analyses to validate
        all_types = ['glm_session', 'glm_subject', 'mvpa', 'correlations', 'figures']
        self.analysis_types = analysis_types or all_types

        # Results storage
        self.results: Dict[str, ValidationResult] = {}

    def log(self, msg: str, level: str = "info"):
        """
        Log message via logger or print.

        Args:
            msg: Message to log
            level: Log level (info, warning, error, debug)
        """
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(msg)

    def get_subject_sessions(self, subject: str) -> List[str]:
        """
        Get all sessions for a subject from raw BIDS data.

        Args:
            subject: Subject ID (e.g., 'sub-01')

        Returns:
            List of session IDs (e.g., ['ses-001', 'ses-002', ...])
        """
        return get_sessions_from_raw_data(self.data_path, subject)

    def validate_glm_session_level(self) -> ValidationResult:
        """
        Validate session-level GLM outputs.

        Returns:
            ValidationResult for session-level GLM
        """
        result = ValidationResult("GLM Session-Level")

        for subject in self.subjects:
            sessions = self.get_subject_sessions(subject)

            for session in sessions:
                for condition in self.conditions:
                    # Check z-map
                    z_map_dir = op.join(
                        self.data_path, "processed", "session-level",
                        subject, session, "z_maps"
                    )
                    base_name = f"{subject}_{session}_task-shinobi_contrast-{condition}"
                    z_map_file = op.join(z_map_dir, f"{base_name}_stat-z.nii.gz")

                    result.add_expected(z_map_file)

                    if self.check_integrity and op.exists(z_map_file):
                        is_valid, error = validate_nifti(z_map_file, check_integrity=True)
                        if not is_valid:
                            result.add_error(z_map_file, error)

                    # Check beta map
                    beta_map_dir = op.join(
                        self.data_path, "processed", "session-level",
                        subject, session, "beta_maps"
                    )
                    beta_map_file = op.join(beta_map_dir, f"{base_name}_stat-beta.nii.gz")
                    result.add_expected(beta_map_file)

        return result

    def validate_glm_subject_level(self) -> ValidationResult:
        """
        Validate subject-level GLM outputs.

        Returns:
            ValidationResult for subject-level GLM
        """
        result = ValidationResult("GLM Subject-Level")

        for subject in self.subjects:
            for condition in self.conditions:
                # Check z-map
                z_map_dir = op.join(
                    self.data_path, "processed", "subject-level",
                    subject, "z_maps"
                )
                base_name = f"{subject}_task-shinobi_contrast-{condition}"
                z_map_file = op.join(z_map_dir, f"{base_name}_stat-z.nii.gz")

                result.add_expected(z_map_file)

                if self.check_integrity and op.exists(z_map_file):
                    is_valid, error = validate_nifti(z_map_file, check_integrity=True)
                    if not is_valid:
                        result.add_error(z_map_file, error)

                # Check beta map
                beta_map_dir = op.join(
                    self.data_path, "processed", "subject-level",
                    subject, "beta_maps"
                )
                beta_map_file = op.join(beta_map_dir, f"{base_name}_stat-beta.nii.gz")
                result.add_expected(beta_map_file)

        return result

    def validate_mvpa(self, screening: int = 20, expected_permutations: int = 1000) -> ValidationResult:
        """
        Validate MVPA outputs including permutation tests.

        Args:
            screening: Screening percentile used (default: 20)
            expected_permutations: Expected number of permutations (default: 1000)

        Returns:
            ValidationResult for MVPA
        """
        result = ValidationResult(f"MVPA (screening={screening})")

        mvpa_dir = op.join(self.data_path, "processed", f"mvpa_results_s{screening}")

        for subject in self.subjects:
            # Main decoder
            decoder_file = op.join(mvpa_dir, f"{subject}_decoder.pkl")
            result.add_expected(decoder_file)

            if self.check_integrity and op.exists(decoder_file):
                is_valid, error = validate_pickle(decoder_file, check_integrity=True)
                if not is_valid:
                    result.add_error(decoder_file, error)

            # Weight maps directory
            weight_map_dir = op.join(mvpa_dir, "weight_maps")
            if op.exists(weight_map_dir):
                # Check for weight maps - one per class
                # Note: Not all conditions may be used in MVPA classification
                for condition in self.conditions:
                    weight_file = op.join(weight_map_dir, f"{subject}_{condition}_weights.nii.gz")
                    if op.exists(weight_file):
                        result.add_expected(weight_file)
                        if self.check_integrity:
                            is_valid, error = validate_nifti(weight_file, check_integrity=True)
                            if not is_valid:
                                result.add_error(weight_file, error)

            # Permutation results
            perm_dir = op.join(mvpa_dir, f"{subject}_permutations")
            if op.exists(perm_dir):
                # Count permutation files
                import glob
                perm_files = glob.glob(op.join(perm_dir, "perm_*.pkl"))

                if perm_files:
                    # Store permutation info
                    if not hasattr(result, 'extra_info'):
                        result.extra_info = {}
                    if 'permutation_counts' not in result.extra_info:
                        result.extra_info['permutation_counts'] = {}

                    result.extra_info['permutation_counts'][subject] = len(perm_files)

                    # Log permutation status
                    self.log(f"  {subject}: Found {len(perm_files)} permutation files", "info")

                    # Each permutation file typically contains 10 permutations
                    # So we expect ~100 files for 1000 permutations
                    expected_files = expected_permutations // 10
                    if len(perm_files) < expected_files:
                        self.log(f"    ⚠ Expected ~{expected_files} files for {expected_permutations} permutations", "warning")

            # Aggregated permutation results
            perm_results_file = op.join(mvpa_dir, f"{subject}_permutation_results.pkl")
            if op.exists(perm_results_file):
                result.add_expected(perm_results_file)
                if self.check_integrity:
                    is_valid, error = validate_pickle(perm_results_file, check_integrity=True)
                    if not is_valid:
                        result.add_error(perm_results_file, error)

        return result

    def validate_correlations(self) -> ValidationResult:
        """
        Validate correlation analysis outputs.

        Checks for correlation matrix file existence and validates its completeness
        by loading the matrix and checking which beta map pairs have been computed.

        Returns:
            ValidationResult for correlations
        """
        result = ValidationResult("Correlations - Beta Maps")

        # Beta correlations
        beta_corr_file = op.join(
            self.data_path, "processed", "beta_maps_correlations.pkl"
        )
        result.add_expected(beta_corr_file)

        if not op.exists(beta_corr_file):
            self.log("Beta correlation matrix not found - no correlations computed yet", "warning")
            return result

        # Load and validate correlation matrix
        self.log("Loading correlation matrix to check completeness...", "info")
        corr_data = load_correlation_matrix(beta_corr_file)

        if corr_data is None:
            result.add_error(beta_corr_file, "Failed to load correlation matrix")
            return result

        # Validate matrix completeness
        stats, missing_maps = validate_correlation_matrix(
            corr_data, self.subjects, self.conditions
        )

        # Report statistics
        self.log(f"  Total maps in matrix: {stats['total_maps_in_matrix']}", "info")
        self.log(f"  Shinobi session-level maps: {stats['shinobi_session_maps']}", "info")
        self.log(f"  Possible correlation pairs: {stats['total_possible_pairs']}", "info")
        self.log(f"  Computed pairs: {stats['computed_pairs']}", "info")
        self.log(f"  Missing pairs: {stats['missing_pairs']}", "info")
        self.log(f"  Matrix completion: {stats['matrix_completion_rate']:.1f}%", "info")

        # Store stats in result for reporting
        if not hasattr(result, 'extra_info'):
            result.extra_info = {}
        result.extra_info['matrix_stats'] = stats
        result.extra_info['missing_map_combos'] = missing_maps

        # Check matrix completeness
        if stats['missing_pairs'] > 0:
            self.log(f"  ⚠ Correlation matrix is incomplete: {stats['missing_pairs']} pairs missing", "warning")

        # Cross-validate: check that session-level beta maps in matrix actually exist
        self.log("Cross-validating correlation matrix against actual beta maps...", "info")
        missing_beta_files = self._cross_validate_correlations(corr_data)

        if missing_beta_files:
            self.log(f"  ⚠ Found {len(missing_beta_files)} beta maps in correlation matrix that don't exist on disk", "warning")
            result.extra_info['missing_beta_files'] = missing_beta_files[:10]  # Store first 10

        return result

    def _cross_validate_correlations(self, corr_data: Dict) -> List[str]:
        """
        Cross-validate correlation matrix against actual beta map files.

        Checks that beta maps listed in the correlation matrix actually exist on disk.

        Args:
            corr_data: Correlation matrix dictionary

        Returns:
            List of beta map file paths that are in matrix but don't exist
        """
        missing_files = []

        map_subjects = corr_data.get('subj', [])
        map_sessions = corr_data.get('ses', [])
        map_conditions = corr_data.get('cond', [])
        map_sources = corr_data.get('source', [])

        for i in range(len(map_subjects)):
            source = map_sources[i]

            # Only check shinobi session-level maps (not HCP)
            if source == 'session-level':
                subj = map_subjects[i]
                ses = map_sessions[i]
                cond = map_conditions[i]

                # Construct expected beta map path
                beta_map_file = op.join(
                    self.data_path, "processed", "session-level",
                    subj, ses, "beta_maps",
                    f"{subj}_{ses}_task-shinobi_contrast-{cond}_stat-beta.nii.gz"
                )

                if not op.exists(beta_map_file):
                    missing_files.append(beta_map_file)

        return missing_files

    def validate_figures(self) -> ValidationResult:
        """
        Validate visualization outputs (figures and reports).

        Checks for key visualization files including GLM reports,
        correlation plots, and subject-level visualizations.

        Returns:
            ValidationResult for figures
        """
        result = ValidationResult("Figures & Visualizations")

        # Get FIG_PATH from config
        fig_path = config.FIG_PATH

        # 1. Beta correlation plots
        beta_corr_plot = op.join(fig_path, "corrmats_withconstant", "ses-level_corrmat.png")
        if op.exists(beta_corr_plot):
            result.add_expected(beta_corr_plot)
        else:
            self.log(f"  Beta correlation plot not found", "debug")

        # 2. Subject-level condition plots
        for subject in self.subjects:
            for condition in self.conditions:
                # Main z-map plot
                zmap_plot = op.join(
                    fig_path, "full_zmap_plot", subject, condition,
                    f"{subject}_{condition}.png"
                )
                result.add_expected(zmap_plot)

        # 3. Annotation panels
        annotation_dir = op.join(fig_path, "full_zmap_plot", "annotations")
        if op.exists(annotation_dir):
            # Check for key annotation files
            for condition in self.conditions:
                annotation_file = op.join(annotation_dir, f"annotation_panel_{condition}.png")
                if op.exists(annotation_file):
                    result.add_expected(annotation_file)

        # 4. GLM reports (HTML)
        for level in ['run-level', 'session-level', 'subject-level']:
            report_dir = op.join(fig_path, level, "reports")
            if op.exists(report_dir):
                import glob
                html_reports = glob.glob(op.join(report_dir, "**/*.html"), recursive=True)
                for report in html_reports:
                    result.add_expected(report)

        # 5. MVPA confusion matrices
        mvpa_fig_dir = op.join(fig_path, "mvpa_confusion_matrices")
        if op.exists(mvpa_fig_dir):
            import glob
            confusion_plots = glob.glob(op.join(mvpa_fig_dir, "*.png"))
            for plot in confusion_plots:
                result.add_expected(plot)

        # Store summary stats
        if not hasattr(result, 'extra_info'):
            result.extra_info = {}

        result.extra_info['figure_counts'] = {
            'subject_level_zmaps': sum(1 for f in result.missing + [f for f in [] if result.expected > 0]
                                      if 'full_zmap_plot' in f and f.endswith('.png')),
            'beta_correlation_plots': 1 if op.exists(beta_corr_plot) else 0,
        }

        return result

    def run_validation(self) -> Dict[str, ValidationResult]:
        """
        Run all requested validations.

        Returns:
            Dictionary of analysis_type -> ValidationResult
        """
        self.log("=" * 80)
        self.log("SHINOBI_FMRI PIPELINE VALIDATION")
        self.log("=" * 80)
        self.log(f"Data path: {self.data_path}")
        self.log(f"Subjects: {', '.join(self.subjects)}")
        self.log(f"Analysis types: {', '.join(self.analysis_types)}")
        self.log(f"Integrity check: {'ENABLED' if self.check_integrity else 'DISABLED'}")
        self.log("=" * 80)

        # Run validations
        if 'glm_session' in self.analysis_types:
            self.log("\n[1/5] Validating GLM session-level outputs...")
            self.results['glm_session'] = self.validate_glm_session_level()

        if 'glm_subject' in self.analysis_types:
            self.log("\n[2/5] Validating GLM subject-level outputs...")
            self.results['glm_subject'] = self.validate_glm_subject_level()

        if 'mvpa' in self.analysis_types:
            self.log("\n[3/5] Validating MVPA outputs...")
            self.results['mvpa'] = self.validate_mvpa()

        if 'correlations' in self.analysis_types:
            self.log("\n[4/5] Validating correlation outputs...")
            self.results['correlations'] = self.validate_correlations()

        if 'figures' in self.analysis_types:
            self.log("\n[5/5] Validating figures and visualizations...")
            self.results['figures'] = self.validate_figures()

        return self.results

    def print_summary(self):
        """Print validation summary to console (always displayed)."""
        # Always print to console regardless of log level
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        for analysis_type, result in self.results.items():
            print(f"\n{result.category}:")
            print(f"  Expected: {result.expected}")
            print(f"  Found:    {result.found}")
            print(f"  Missing:  {len(result.missing)}")
            print(f"  Complete: {result.completion_rate:.1f}%")

            if result.errors:
                print(f"  Errors:   {len(result.errors)}")

            # Show extra info for correlations and MVPA
            if hasattr(result, 'extra_info'):
                if 'matrix_stats' in result.extra_info:
                    stats = result.extra_info['matrix_stats']
                    print(f"  Matrix Pairs: {stats['computed_pairs']}/{stats['total_possible_pairs']} ({stats['matrix_completion_rate']:.1f}%)")

        # Overall stats
        total_expected = sum(r.expected for r in self.results.values())
        total_found = sum(r.found for r in self.results.values())
        total_missing = sum(len(r.missing) for r in self.results.values())

        print("\n" + "=" * 80)
        print("OVERALL:")
        print(f"  Expected: {total_expected}")
        print(f"  Found:    {total_found}")
        print(f"  Missing:  {total_missing}")
        if total_expected > 0:
            overall_completion = (total_found / total_expected) * 100
            print(f"  Complete: {overall_completion:.1f}%")
        print("=" * 80)

    def save_detailed_report(self, output_path: str):
        """
        Save detailed validation report to JSON file.

        Args:
            output_path: Path to save JSON report
        """
        report = {
            'data_path': self.data_path,
            'subjects': self.subjects,
            'analysis_types': self.analysis_types,
            'check_integrity': self.check_integrity,
            'results': {}
        }

        for analysis_type, result in self.results.items():
            result_dict = {
                'category': result.category,
                'expected': result.expected,
                'found': result.found,
                'missing_count': len(result.missing),
                'completion_rate': result.completion_rate,
                'missing_files': result.missing,
                'errors': [{'file': f, 'error': e} for f, e in result.errors]
            }

            # Add extra_info if present (e.g., correlation matrix stats)
            if hasattr(result, 'extra_info'):
                result_dict['extra_info'] = result.extra_info

            report['results'][analysis_type] = result_dict

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.log(f"\nDetailed report saved to: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate shinobi_fmri pipeline outputs"
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Specific subject to validate (e.g., sub-01). Default: all subjects'
    )
    parser.add_argument(
        '--analysis-type',
        type=str,
        choices=['glm_session', 'glm_subject', 'mvpa', 'correlations', 'figures', 'all'],
        default='all',
        help='Type of analysis to validate. Default: all'
    )
    parser.add_argument(
        '--check-integrity',
        action='store_true',
        help='Check file integrity (load and validate contents). Slower but thorough.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save detailed JSON report (optional)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (e.g., -v for INFO, -vv for DEBUG)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Directory for log files'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = AnalysisLogger(
        log_name="validation",
        log_dir=args.log_dir,
        verbosity=log_level
    )

    # Determine subjects
    subjects = [args.subject] if args.subject else None

    # Determine analysis types
    if args.analysis_type == 'all':
        analysis_types = None  # Will default to all
    else:
        analysis_types = [args.analysis_type]

    # Create validator
    validator = PipelineValidator(
        data_path=config.DATA_PATH,
        subjects=subjects,
        analysis_types=analysis_types,
        check_integrity=args.check_integrity,
        logger=logger
    )

    try:
        # Run validation
        validator.run_validation()

        # Print summary
        validator.print_summary()

        # Save detailed report if requested
        if args.output:
            validator.save_detailed_report(args.output)

    finally:
        logger.close()


if __name__ == '__main__':
    main()
