#!/bin/bash

# Create and setup virtual environment for PyCuVSLAM tools
# Usage: ./create_env.sh

set -e  # Exit on error

echo "Creating virtual environment..."
python3 -m venv .env

echo "Activating virtual environment..."
source .env/bin/activate

echo "Installing pycuvslam-tools and its dependencies..."
pip install --upgrade pip

SCRIPT_FULL_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pip install -e "$SCRIPT_FULL_PATH"

echo ""
echo "✓ Virtual environment created and dependencies installed!"
echo ""
echo "To activate the environment, run:"
echo "  source .env/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
