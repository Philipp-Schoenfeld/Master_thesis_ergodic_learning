def fix(fpath, search_str):
    with open(fpath, 'r') as f:
        c = f.read()
    import re
    # We want to replace the whole LATEST_CKPT=... line
    c = re.sub(r'LATEST_CKPT=\$\(ls -t .*?\)', f'LATEST_CKPT=$(ls -t checkpoints/*{search_str}*.pt 2>/dev/null | head -1)', c)
    with open(fpath, 'w') as f:
        f.write(c)

fix('thesis_architecture/run_job_ergodic.bash', 'flow_matching_ergodic_date_*_S256_nxi25_D384_C100_flip0.0_ep')
fix('thesis_architecture/run_job_ergodic_50c.bash', 'flow_matching_ergodic_date_*_S256_nxi25_D384_C50_flip0.0_ep')

import os
if os.path.exists('thesis_architecture/run_job_waypoint.bash'):
    fix('thesis_architecture/run_job_waypoint.bash', 'flow_matching_waypoint_date_*_ep')

