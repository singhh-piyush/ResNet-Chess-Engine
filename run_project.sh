#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 7860
