#!/usr/bin/env python3
import os
import subprocess
import threading
import sys

def run_script(script, log_file, position):
    env = os.environ.copy()
    env["TQDM_POSITION"] = str(position)
    # We redirect stdout to log, but let stderr (tqdm) go to the terminal
    with open(log_file, "w") as f:
        subprocess.run(["python3", script], stdout=f, stderr=sys.stderr, env=env)

def main():
    shape = sys.argv[1] if len(sys.argv) > 1 else "N"
    proj = sys.argv[2] if len(sys.argv) > 2 else "plane"
    
    # We must set this so the scripts pick it up
    os.environ["TARGET_SHAPE"] = shape
    os.environ["TARGET_PROJECTION"] = proj

    print(f"=== Starting 3D runs for shape '{shape}' with projection '{proj}' ===")
    print(f"Logs will be saved to tsvec_3d_{shape}_{proj}.log and svgd_bspline_3d_{shape}_{proj}.log")
    
    # Print a few newlines so tqdm has space to draw the bars without overwriting our text
    print("\n\n")

    # Run both scripts in parallel threads
    t1 = threading.Thread(target=run_script, args=("tsvec_3d.py", f"tsvec_3d_{shape}_{proj}.log", 0))
    t2 = threading.Thread(target=run_script, args=("svgd_bspline_3d.py", f"svgd_bspline_3d_{shape}_{proj}.log", 1))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    
    print("\n=== All done ===")

if __name__ == "__main__":
    main()
