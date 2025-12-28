# Changelog

All notable changes to the shinobi_fmri project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed
- **Provenance tracking**: Fixed git commit hash always being null in metadata files. The repo root detection now falls back to searching from the current working directory when `.git` is not found relative to the script path. This ensures git information is captured correctly when running via SLURM or when using virtual environments located outside the repository.

