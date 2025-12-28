#!/bin/bash
# Utility functions for renaming SLURM logs based on job exit status
# This script should be sourced by SLURM job scripts: source rename_logs_on_exit.sh

# Global variable to track termination reason
JOB_STATUS=""

# Function to rename log files based on status
# Handles multiple log file naming patterns:
# - Standard: {job_name}_{job_id}.out
# - Array jobs: {job_name}_{array_id}_{task_id}.out
# - Chunk script: {job_name}_{job_id}.out (no subdirectory)
rename_logs() {
    local status="$1"
    local job_name="${SLURM_JOB_NAME}"
    local job_id="${SLURM_JOB_ID}"

    # Determine log file pattern based on whether this is an array job
    local file_pattern
    if [ -n "$SLURM_ARRAY_JOB_ID" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        # Array job: use %A_%a pattern (array_job_id_task_id)
        file_pattern="${job_name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    else
        # Standard job: use %j pattern (job_id)
        file_pattern="${job_name}_${job_id}"
    fi

    # Try both with and without subdirectory
    local log_locations=(
        "${LOGS_DIR}/slurm/${job_name}"     # Standard: logs/slurm/{job_name}/
        "${LOGS_DIR}/slurm"                  # Chunk style: logs/slurm/
    )

    for log_dir in "${log_locations[@]}"; do
        local old_out="${log_dir}/${file_pattern}.out"
        local old_err="${log_dir}/${file_pattern}.err"
        local new_out="${log_dir}/${file_pattern}_${status}.out"
        local new_err="${log_dir}/${file_pattern}_${status}.err"

        # Rename if files exist
        if [ -f "$old_out" ]; then
            mv "$old_out" "$new_out" 2>/dev/null && break
        fi
    done

    # Same for .err files
    for log_dir in "${log_locations[@]}"; do
        local old_err="${log_dir}/${file_pattern}.err"
        local new_err="${log_dir}/${file_pattern}_${status}.err"

        if [ -f "$old_err" ]; then
            mv "$old_err" "$new_err" 2>/dev/null && break
        fi
    done
}

# Trap handler for job termination signals
handle_termination() {
    local signal="$1"

    case "$signal" in
        TERM|INT)
            JOB_STATUS="cancelled"
            ;;
        USR1)
            # SLURM sends SIGUSR1 before timeout
            JOB_STATUS="timeout"
            ;;
        *)
            JOB_STATUS="terminated"
            ;;
    esac

    rename_logs "$JOB_STATUS"
    exit 143  # 128 + 15 (SIGTERM)
}

# Set up traps for termination signals
# SIGTERM: normal cancellation
# SIGINT: interactive cancellation
# SIGUSR1: timeout warning (some SLURM configs send this before SIGTERM on timeout)
trap 'handle_termination TERM' TERM
trap 'handle_termination INT' INT
trap 'handle_termination USR1' USR1

# Function to wrap command execution and rename logs based on exit code
# Usage: run_and_rename_logs <command> [args...]
run_and_rename_logs() {
    # Run the command
    "$@"
    local exit_code=$?

    # Determine status based on exit code
    if [ $exit_code -eq 0 ]; then
        JOB_STATUS="finished"
    else
        JOB_STATUS="failed"
    fi

    # Rename logs
    rename_logs "$JOB_STATUS"

    # Return original exit code
    return $exit_code
}
