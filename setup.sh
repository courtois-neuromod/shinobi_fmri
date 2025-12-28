#!/bin/bash
#
# Automated setup script for shinobi_fmri pipeline
#
# This script:
#   1. Checks Python version requirements
#   2. Creates a virtual environment
#   3. Installs all dependencies with pinned versions
#   4. Sets up configuration template
#   5. Verifies installation
#
# Usage:
#   bash setup.sh
#   or
#   ./setup.sh

set -e  # Exit on error

echo "=========================================="
echo "shinobi_fmri Pipeline Setup"
echo "=========================================="
echo ""

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function echo_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

function echo_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function echo_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# 1. Check Python version
echo_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo_error "Python 3.8+ required. Found: $PYTHON_VERSION"
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi
echo_success "Python version $PYTHON_VERSION OK (>= 3.8 required)"

# 2. Create virtual environment
echo ""
echo_info "Creating virtual environment in ./env/..."
if [ -d "env" ]; then
    echo "Virtual environment already exists at ./env/"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf env
        python3 -m venv env
        echo_success "Virtual environment recreated"
    else
        echo_info "Using existing virtual environment"
    fi
else
    python3 -m venv env
    echo_success "Virtual environment created"
fi

# 3. Activate virtual environment
echo ""
echo_info "Activating virtual environment..."
source env/bin/activate
echo_success "Virtual environment activated"

# 4. Upgrade pip
echo ""
echo_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo_success "pip upgraded to $(pip --version | awk '{print $2}')"

# 5. Install dependencies
echo ""
echo_info "Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo_success "All dependencies installed"

# 6. Set up configuration
echo ""
echo_info "Setting up configuration..."
if [ -f "config.yaml" ]; then
    echo "config.yaml already exists"
    read -p "Do you want to view it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cat config.yaml
    fi
else
    if [ -f "config.yaml.template" ]; then
        cp config.yaml.template config.yaml
        echo_success "config.yaml created from template"
        echo ""
        echo_error "IMPORTANT: You must edit config.yaml and replace all <PLACEHOLDER> values!"
        echo "          Edit the file with your actual paths before running analysis tasks."
        echo ""
        echo "          Key placeholders to replace:"
        echo "            - paths.data: /your/path/to/shinobi_data"
        echo "            - python.local_bin: /your/path/to/python (or just 'python')"
        echo "            - python.slurm_bin: /your/path/to/python on HPC"
    else
        echo_error "config.yaml.template not found!"
        exit 1
    fi
fi

# 7. Verify installation
echo ""
echo_info "Verifying installation..."
python -c "import shinobi_fmri; import nilearn; import numpy; import pandas" 2>/dev/null
if [ $? -eq 0 ]; then
    echo_success "Installation verification successful"
else
    echo_error "Installation verification failed"
    echo "Some packages may not have installed correctly."
    exit 1
fi

# 8. Display package versions
echo ""
echo_info "Installed package versions:"
pip list | grep -E "(nilearn|numpy|pandas|scipy|matplotlib|invoke)" | column -t

# 9. Final instructions
echo ""
echo "=========================================="
echo_success "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml and replace all <PLACEHOLDER> values with your paths"
echo "  2. Activate the environment: source env/bin/activate"
echo "  3. Check configuration: invoke info"
echo "  4. List available tasks: invoke --list"
echo "  5. See TASKS.md for detailed usage examples"
echo ""
echo "For help:"
echo "  - README.md: Installation and overview"
echo "  - TASKS.md: Detailed task documentation"
echo ""
