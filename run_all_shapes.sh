#!/bin/bash
set -e

export PYTHONPATH=$(pwd)/src:$(pwd)

SHAPES=("N" "H" "II")
SCRIPTS=(
    "CE_Ergodic/ce_ergodic_2d.py"
    "LB_Ergodic/lb_ergodic_2d.py"
    "SigKernel_CMA/sv_cma_es_2d.py"
    "SE3_SVGD/tsvec_2d.py"
    "SE3_SVGD/svgd_bspline_2d.py"
    "Stein_Flow_matching/flow_matching_2d.py"
)

for script in "${SCRIPTS[@]}"; do
    for shape in "${SHAPES[@]}"; do
        echo "Running $script with shape $shape"
        TARGET_SHAPE=$shape python3 "$script"
    done
done
