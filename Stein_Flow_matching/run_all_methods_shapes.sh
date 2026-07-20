#!/bin/bash
set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$(pwd)/../src:$(pwd)/..

SHAPES=("N" "H" "II")
METHODS=(
    "flow_matching_chebyshev_2d.py"
    "flow_matching_rbf_2d.py"
    "flow_matching_bspline_2d.py"
)

# You can adjust N_ITERS here if 800000 takes too long. Using default for now.
# export N_ITERS=100000 

for shape in "${SHAPES[@]}"; do
    for method in "${METHODS[@]}"; do
        log_file="${method%.py}_${shape}.log"
        echo "Running $method with shape $shape... Logging to $log_file"
        TARGET_SHAPE=$shape python3 -u "$method" > "$log_file" 2>&1
    done
done

echo "All flow matching methods and shapes completed!"
