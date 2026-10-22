"""Independent validation of ergodic trajectory solvers (Sun FM-Stein, Adam-SVGD, TSVEC-SVGD).

`ergodicity_metrics` -- metrics that do NOT depend on the Fourier metric the solvers optimise
`targets`            -- canonical target densities with known answers
`validate_solvers`   -- test bench: runs every solver variant of the smoothness pipeline on the
                        canonical targets and checks the results against pass/fail criteria
"""
