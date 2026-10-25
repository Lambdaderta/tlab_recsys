#!/bin/bash
# Task 1: OpenFE-style feature generation and the graph model (2K users; a laptop is enough, ~10 min).
set -e
cd "$(dirname "$0")/../task1_feature_engineering"
uv run python openfe.py   # -> out/openfe_report.json
uv run python graph.py    # -> out/graph_report.json (uses the features OpenFE selected)
