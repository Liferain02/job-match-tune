#!/usr/bin/env bash
set -euo pipefail
conda run -n tune-demo python -m pip install -r requirements.txt
conda run -n tune-demo python -m pip install -e .
