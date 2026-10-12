#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export DISPLAY=${DISPLAY:-:0}
exec .venv/bin/python app.py
