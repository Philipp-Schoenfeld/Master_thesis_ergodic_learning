#!/bin/bash
# Wrapper to run the 3D SVGD methods in parallel with progress bars
# Usage: ./run_3d.sh [SHAPE] [PROJECTION]  (default: N plane)

DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$DIR/run_3d.py" "$@"

echo "=== All done ==="
