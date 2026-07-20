#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export N_ITERS=500
echo "Running Chebyshev..."
python3 -u flow_matching_chebyshev_2d.py > chebyshev.log 2>&1
echo "Running RBF..."
python3 -u flow_matching_rbf_2d.py > rbf.log 2>&1
echo "Running BSpline..."
python3 -u flow_matching_bspline_2d.py > bspline.log 2>&1
echo "Done!"
