#!/bin/sh

# Description:
# This script builds documentation for the Python cuvslam package.
# It generates HTML documentation using Sphinx.
# It requires the following Python packages to be installed
# in the current Python environment:
# - sphinx
# - sphinx-rtd-theme
# It also requires the Python cuvslam package to be installed.
#
# Usage:
#   ./build_docs.sh --build_dir=BUILD_DIR --src_dir=SRC_DIR
#
# Arguments:
#   --build_dir: CMake build directory
#   --src_dir: Source directory

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
BUILD_DIR=""
SRC_DIR=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --build_dir=*)
      BUILD_DIR="${1#*=}"
      ;;
    --src_dir=*)
      SRC_DIR="${1#*=}"
      ;;
    *)
      echo "Error: Unknown argument: $1"
      echo "Usage: $0 --build_dir=BUILD_DIR --src_dir=SRC_DIR"
      exit 1
      ;;
  esac
  shift
done

# Check if required arguments are provided
if [ -z "$BUILD_DIR" ] || [ -z "$SRC_DIR" ]; then
  echo "Error: Both --build_dir and --src_dir are required"
  echo "Usage: $0 --build_dir=BUILD_DIR --src_dir=SRC_DIR"
  exit 1
fi

# Check if directories exist
if [ ! -d "$BUILD_DIR" ]; then
  echo "Error: Build directory '$BUILD_DIR' does not exist."
  exit 1
fi
if [ ! -d "$SRC_DIR" ]; then
  echo "Error: Source directory '$SRC_DIR' does not exist."
  exit 1
fi

# Check if required Python packages are installed
if ! python3 -c "import sphinx, sphinx_rtd_theme" 2>/dev/null; then
  echo "Error: Required Python packages (sphinx, sphinx-rtd-theme) not found."
  echo "Please install them, e.g. using: python3 -m pip install --user sphinx sphinx-rtd-theme"
  exit 1
fi

# Check that Python cuvslam package is installed
python3 -c "import cuvslam" 2>/dev/null || {
  echo "Error: Python cuvslam package is not installed in the current Python environment."
  exit 1
}

# Create documentation directory in build directory
DOCS_BUILD_DIR="$BUILD_DIR/python_docs"
mkdir -p "$DOCS_BUILD_DIR"

# Build documentation
python3 -m sphinx -b html "$SRC_DIR" "$DOCS_BUILD_DIR/html"

echo "You can open $(realpath "$DOCS_BUILD_DIR/html/index.html") in your browser to view documentation."
