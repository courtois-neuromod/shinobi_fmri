# shinobi_fmri

fMRI analysis pipeline for the [cneuromod.shinobi](https://github.com/courtois-neuromod/shinobi_access) dataset.

## Installation

This package requires Python 3.8+.

```bash
# Clone the repository
git clone https://github.com/courtois-neuromod/shinobi_fmri.git
cd shinobi_fmri

# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate

# Install the package and dependencies
pip install -e .
```

## Configuration

The pipeline uses environment variables for configuration. You can set them in your shell or create a `.env` file in the project root:

```bash
# .env file example
SHINOBI_DATA_PATH=/path/to/data
SHINOBI_PYTHON_BIN=/path/to/python
SHINOBI_SLURM_PYTHON_BIN=/path/to/python_on_slurm_nodes
```

| Variable | Description | Default |
|----------|-------------|---------|
| `SHINOBI_DATA_PATH` | Path to the root data directory containing `shinobi.fmriprep` | `/home/hyruuk/scratch/data` |
| `SHINOBI_PYTHON_BIN` | Python interpreter for local execution | `python` |
| `SHINOBI_SLURM_PYTHON_BIN` | Python interpreter for SLURM jobs | `python` |

## Usage

This project uses `invoke` to manage analysis tasks.

List all available tasks:
```bash
invoke --list
```

Run a GLM analysis locally for a specific subject and session:
```bash
invoke glm.run-level --subject sub-01 --session ses-001
```

For more detailed usage instructions, including batch processing and visualization, see [TASKS_USAGE.md](TASKS_USAGE.md).

## Project Structure

- `shinobi_fmri/`: Main package source code
  - `glm/`: General Linear Model analysis scripts
  - `mvpa/`: Multi-Voxel Pattern Analysis scripts
  - `visualization/`: Plotting and reporting tools
- `slurm/`: SLURM batch submission scripts
- `tasks.py`: Task definitions for `invoke` automation