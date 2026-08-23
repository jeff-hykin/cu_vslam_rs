#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -e "${SCRIPT_DIR}/../python_tools"
