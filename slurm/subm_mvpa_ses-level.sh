#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Arguments from task launcher:
# $1 = subject (e.g., sub-01)
# $2 = screening (percentile, default: 20)
# $3 = n_jobs (CPUs, default: 40)

SUBJECT=$1
SCREENING=${2:-20}
N_JOBS=${3:-40}

# Load configuration from config.yaml
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/load_config.sh"

# Create log directory
mkdir -p "$LOGS_DIR/slurm/shi_mvpa_seslvl"

echo "=========================================="
echo "MVPA Session-Level Decoder"
echo "=========================================="
echo "Subject:   $SUBJECT"
echo "Screening: $SCREENING%"
echo "CPUs:      $N_JOBS"
echo "Python:    $PYTHON_BIN"
echo "Data path: $DATA_PATH"
echo "=========================================="

# Run MVPA decoder (no permutations)
"$PYTHON_BIN" "$SCRIPTS_DIR/mvpa/compute_mvpa.py" \
    --subject $SUBJECT \
    --screening $SCREENING \
    --n-jobs $N_JOBS \
    -v

echo "=========================================="
echo "Decoder completed"
echo "=========================================="
