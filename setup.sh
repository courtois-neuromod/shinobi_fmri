#!/bin/bash
# ==============================================================================
# Shinobi fMRI Setup Script
# ==============================================================================
# Sets up the shinobi_fmri environment: venv, dependencies, config.
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   --python      Specify Python executable (default: auto-detect)
#   --force       Force reinstall if venv already exists
#   --help        Show this help message
# ==============================================================================

set -e

PYTHON_CMD=""
FORCE_REINSTALL=false
VENV_DIR="env"

# --- Helpers -----------------------------------------------------------------

print_header() { echo "==================================================================="; echo "$1"; echo "==================================================================="; }
print_success() { echo "[OK] $1"; }
print_error()   { echo "[ERROR] $1"; }
print_warning() { echo "[WARNING] $1"; }
print_info()    { echo "[INFO] $1"; }

show_help() { head -n 14 "$0" | tail -n 8; exit 0; }

# --- Parse arguments ---------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --force)  FORCE_REINSTALL=true; shift ;;
        --help)   show_help ;;
        *)        print_error "Unknown option: $1"; show_help ;;
    esac
done

# --- Python detection --------------------------------------------------------

find_best_python() {
    for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$cmd" &>/dev/null; then
            local vi=$("$cmd" -c 'import sys; print(sys.version_info.major, sys.version_info.minor)' 2>/dev/null)
            local major=$(echo "$vi" | cut -d' ' -f1)
            local minor=$(echo "$vi" | cut -d' ' -f2)
            if [[ "$major" == "3" && "$minor" -ge 9 && "$minor" -le 12 ]]; then
                echo "$cmd"; return 0
            fi
        fi
    done
    return 1
}

# --- Preflight ---------------------------------------------------------------

print_header "Shinobi fMRI Environment Setup"

if [[ -z "$PYTHON_CMD" ]]; then
    PYTHON_CMD=$(find_best_python) || { print_error "No suitable Python found (requires 3.9-3.12)"; exit 1; }
    print_info "Auto-detected Python: $PYTHON_CMD"
fi

PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
print_success "Found Python $PYTHON_VERSION"

if ! "$PYTHON_CMD" -c "import venv" 2>/dev/null; then
    print_error "Python venv module not found (apt install python3-venv)"
    exit 1
fi

# --- Venv --------------------------------------------------------------------

print_header "Setting up virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    if [[ "$FORCE_REINSTALL" == true ]]; then
        rm -rf "$VENV_DIR"
    else
        print_warning "Virtual environment already exists at: $VENV_DIR"
        read -p "Remove and recreate? (y/N): " -n 1 -r; echo
        [[ $REPLY =~ ^[Yy]$ ]] && rm -rf "$VENV_DIR" || print_info "Using existing venv"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel > /dev/null
print_success "Build tools updated"

# --- Install -----------------------------------------------------------------

print_header "Installing shinobi_fmri"

pip install -e .
print_success "Package installed"

# --- Config ------------------------------------------------------------------

print_header "Configuration"

if [[ -f "config.yaml" ]]; then
    print_info "config.yaml already exists (not overwriting)"
else
    if [[ ! -f "config.yaml.template" ]]; then
        print_error "config.yaml.template not found"; exit 1
    fi

    cp config.yaml.template config.yaml
    print_success "config.yaml created from template"
    echo ""

    # Only placeholder: <DATA_PATH>
    echo "Data directory (should contain shinobi/, shinobi.fmriprep/, cneuromod.processed/):"
    read -p "  Enter full path (or press Enter to skip): " DATA_ROOT
    echo ""

    if [[ -n "$DATA_ROOT" ]]; then
        DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
        DATA_ROOT="${DATA_ROOT%/}"
        sed -i "s|<DATA_PATH>|$DATA_ROOT|g" config.yaml
        print_success "Data path set to: $DATA_ROOT"
    else
        print_warning "<DATA_PATH> placeholder left in config.yaml — edit it before running"
    fi
fi

# --- Directories -------------------------------------------------------------

mkdir -p logs reports/figures reports/tables

# --- Verify ------------------------------------------------------------------

print_header "Verifying installation"

python -c "import shinobi_fmri" 2>/dev/null && print_success "shinobi_fmri importable" || print_error "shinobi_fmri cannot be imported"
python -c "import shinobi_fmri.config" 2>/dev/null && print_success "Config loads OK" || print_warning "Config not loadable yet (set <DATA_PATH> first)"
python -c "import invoke" 2>/dev/null && print_success "invoke available" || print_warning "invoke not found"

# --- Done --------------------------------------------------------------------

print_header "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. source env/bin/activate"
echo "  2. Edit config.yaml if needed (set data path)"
echo "  3. invoke info"
echo "  4. invoke --list"
echo ""
print_success "Done!"
