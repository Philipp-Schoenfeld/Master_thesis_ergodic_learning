#!/usr/bin/env python3
import os
import time
import json
import importlib.util
from datetime import datetime
import sys

# Algorithms and their relative paths
ALGORITHMS = {
    "CE_Ergodic": "CE_Ergodic/ce_ergodic_2d.py",
    "LB_Ergodic": "LB_Ergodic/lb_ergodic_2d.py",
    "SE3_SVGD": "SE3_SVGD/tsvec_2d.py",
    "SigKernel_CMA": "SigKernel_CMA/sv_cma_es_2d.py",
    "Stein_Flow_matching": "Stein_Flow_matching/flow_matching_2d.py",
    "OT_CFM": "OT_CFM/ot_cfm_2d.py",
    "Unified_Pipeline": "Unified_Pipeline/unified_pipeline_2d.py"
}

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    base_dir = "/home/philipp/Documents/Uni/Master_thesis"
    os.chdir(base_dir)
    
    # Toggle obstacle experiment
    RUN_OBSTACLE_EXPERIMENT = True
    
    # Create master run folder
    timestamp = datetime.now().strftime("%H-%M_%d-%m")
    folder_name = f"run_{timestamp}_Obstacle" if RUN_OBSTACLE_EXPERIMENT else f"run_{timestamp}"
    run_dir = os.path.join(base_dir, "results", folder_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"============================================================")
    print(f"MASTER BENCHMARK: Starting at {timestamp}")
    print(f"Saving all results to: {run_dir}")
    print(f"============================================================\n")

    master_summary = {}

    for algo_name, rel_path in ALGORITHMS.items():
        print(f"\n[{algo_name}] Loading algorithm...")
        algo_dir = os.path.join(run_dir, algo_name)
        file_path = os.path.join(base_dir, rel_path)
        
        try:
            # Dynamically load the python script
            module_load_name = f"{algo_name}_module"
            module = load_module_from_path(module_load_name, file_path)
            
            # Execute its run_benchmark function
            print(f"[{algo_name}] Executing benchmark (Obstacle: {RUN_OBSTACLE_EXPERIMENT})...")
            algo_results = module.run_benchmark(out_dir=algo_dir, save_npy=True, use_obstacle=RUN_OBSTACLE_EXPERIMENT)
            master_summary[algo_name] = algo_results
            print(f"[{algo_name}] Completed successfully!\n")
            
        except Exception as e:
            print(f"[{algo_name}] FAILED! Error: {e}")
            master_summary[algo_name] = {"error": str(e)}

    # Save the master summary JSON
    summary_path = os.path.join(run_dir, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=4)
        
    print(f"============================================================")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Summary saved to {summary_path}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
