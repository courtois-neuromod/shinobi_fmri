# SLURM Log File Naming with Exit Status

## Overview

All SLURM job scripts automatically rename their log files based on the job's exit status. This makes it easy to identify which jobs succeeded, failed, were cancelled, or timed out.

## Log File Naming Convention

Log files are named with the following pattern:

```
logs/slurm/{job_name}/{job_name}_{job_id}_{status}.{out|err}
```

Where `{status}` can be:
- **`finished`** - Job completed successfully (exit code 0)
- **`failed`** - Job failed with non-zero exit code
- **`cancelled`** - Job was manually cancelled (scancel or Ctrl+C)
- **`timeout`** - Job exceeded time limit

### Examples

```bash
# Successful session-level GLM
logs/slurm/shi_sesslevel/shi_sesslevel_12345678_finished.out
logs/slurm/shi_sesslevel/shi_sesslevel_12345678_finished.err

# Failed MVPA job
logs/slurm/shi_mvpa_seslvl/shi_mvpa_seslvl_87654321_failed.out
logs/slurm/shi_mvpa_seslvl/shi_mvpa_seslvl_87654321_failed.err

# Cancelled correlation job
logs/slurm/shinobi_corrmat/shinobi_corrmat_11223344_cancelled.out

# Timed out visualization
logs/slurm/shi_viz_sublvl/shi_viz_sublvl_55667788_timeout.out
```

## How It Works

1. Each SLURM script sources `slurm/rename_logs_on_exit.sh`
2. The script sets up signal traps to catch cancellations and timeouts
3. After the Python command completes, the log files are renamed based on:
   - Exit code (0 = finished, non-zero = failed)
   - Signal received (SIGTERM/SIGINT = cancelled, SIGUSR1 = timeout)

## Finding Failed Jobs

To quickly identify failed jobs:

```bash
# List all failed jobs
find logs/slurm -name "*_failed.out"

# List all cancelled jobs
find logs/slurm -name "*_cancelled.out"

# List all timed out jobs
find logs/slurm -name "*_timeout.out"

# Count job outcomes
echo "Finished: $(find logs/slurm -name "*_finished.out" | wc -l)"
echo "Failed:   $(find logs/slurm -name "*_failed.out" | wc -l)"
echo "Cancelled: $(find logs/slurm -name "*_cancelled.out" | wc -l)"
echo "Timeout:   $(find logs/slurm -name "*_timeout.out" | wc -l)"
```

## Checking Recent Job Status

Use `sacct` to see job exit codes and correlate with log files:

```bash
# Recent jobs with their status
sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End -S $(date -d '7 days ago' +%Y-%m-%d)

# Failed jobs in last 7 days
sacct -u $USER --state=FAILED --format=JobID,JobName,ExitCode -S $(date -d '7 days ago' +%Y-%m-%d)

# Cancelled jobs
sacct -u $USER --state=CANCELLED --format=JobID,JobName,Start,End -S $(date -d '7 days ago' +%Y-%m-%d)
```

## Implementation Details

- **Array jobs** use pattern: `{job_name}_{array_id}_{task_id}_{status}.out`
- **Signal handling** catches SIGTERM (cancel), SIGINT (interactive cancel), SIGUSR1 (timeout warning)
- **Log location detection** automatically handles both standard (`logs/slurm/{job_name}/`) and flat (`logs/slurm/`) directory structures
- **Graceful degradation** - if renaming fails, logs remain with original names

## Troubleshooting

### Logs not renamed
- Check that `slurm/rename_logs_on_exit.sh` exists and is executable
- Verify the SLURM script sources both `load_config.sh` and `rename_logs_on_exit.sh`
- Ensure `$LOGS_DIR` environment variable is set correctly

### Timeout vs Cancelled
- SLURM may send SIGTERM for both timeout and manual cancellation
- Some clusters send SIGUSR1 before timeout - check your cluster's configuration
- If uncertain, check `sacct` output for the job's State field

### Finding the corresponding job ID
- Log filename contains the job ID: `shi_sesslevel_12345678_finished.out` → job ID is 12345678
- Use: `sacct -j 12345678 --format=JobID,JobName,State,ExitCode,Start,End` to get details
