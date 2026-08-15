import sys, re

def fix_runner(file_path, prefix, include_c=True):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Import datetime
    if 'from datetime import datetime' not in content:
        content = content.replace('import argparse, os, random, sqlite3, sys, math, json',
                                  'import argparse, os, random, sqlite3, sys, math, json\nfrom datetime import datetime')
        content = content.replace('import argparse, os, random, sqlite3, sys, math',
                                  'import argparse, os, random, sqlite3, sys, math\nfrom datetime import datetime')

    # 2. Inject timestamp creation inside run(args)
    if 'args.run_str =' not in content:
        if include_c:
            run_str_def = f'args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")\n    args.run_str = f"flow_matching_{prefix}_{{args.timestamp}}_S{{args.S}}_nxi{{args.nxi}}_D{{args.D}}_C{{args.copies_per_char}}_flip{{args.p_flip}}"'
        else:
            run_str_def = f'args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")\n    args.run_str = f"flow_matching_{prefix}_{{args.timestamp}}_nxi{{args.nxi}}_D{{args.D}}_flip{{args.p_flip}}"'
        
        content = re.sub(r'def run\(args\):\n(.*?print\(f"\\n\{\'=\' \* 70\}"\))',
                         rf'def run(args):\n    if not hasattr(args, "run_str"):\n        {run_str_def}\n\1',
                         content, flags=re.DOTALL)

    # 3. Clean up _save_checkpoint
    content = re.sub(r'run_str\s*=\s*f"[^"]+"\n\s+path\s*=\s*f"\{stem\}_\{run_str\}_ep\{epoch\+1:04d\}\.pt"',
                     r'path = f"{stem}_{args.run_str}_ep{epoch+1:04d}.pt"', content)

    # 4. Clean up _save_viz
    content = re.sub(r'run_str\s*=\s*f"[^"]+"\n\s+tag_str\s*=\s*f"_\{tag\}" if tag else ""\n\s+viz_path\s*=\s*os\.path\.join\(out_dir, f\'[^_]+_\{run_str\}\{tag_str\}_ep\{ep_num:04d\}\.png\'\)',
                     r'tag_str  = f"_{tag}" if tag else ""\n    viz_path = os.path.join(out_dir, f"{args.run_str}{tag_str}_ep{ep_num:04d}.png")', content)
                     
    # 5. Clean up WandB init
    content = re.sub(r'run_str\s*=\s*f"[^"]+"\n\s+wandb\.init\(\n\s+project=args\.wandb_project,\n\s+name=args\.run_name if args\.run_name else run_str,',
                     r'wandb.init(\n                    project=args.wandb_project,\n                    name=args.run_name if args.run_name else args.run_str,', content)

    # 6. Clean up final save
    content = re.sub(r'run_str\s*=\s*f"[^"]+"\n\s+final_save_path\s*=\s*f"\{stem\}_\{run_str\}_final\.pt"',
                     r'final_save_path = f"{stem}_{args.run_str}_final.pt"', content)

    # 7. Clean up final visualisations
    # "char_train_generation_...png" -> "{args.run_str}_train.png"
    content = re.sub(r'run_str\s*=\s*f"[^"]+"\n\s+visualise_set\((.*?),\n\s*os\.path\.join\(out_dir, f\'[^\']+\'\)',
                     r'visualise_set(\1,\n        os.path.join(out_dir, f"{args.run_str}_train.png")', content, flags=re.DOTALL, count=1)
    
    content = re.sub(r'visualise_set\((.*?),\n\s*os\.path\.join\(out_dir, f\'[^\']+\'\)',
                     r'visualise_set(\1,\n        os.path.join(out_dir, f"{args.run_str}_holdout.png")', content, flags=re.DOTALL)

    with open(file_path, 'w') as f:
        f.write(content)

fix_runner('thesis_architecture/flow_matching_runner_ergodic.py', 'ergodic', include_c=True)
fix_runner('thesis_architecture/flow_matching_runner_spectral.py', 'spectral_outline', include_c=True)
fix_runner('thesis_architecture/flow_matching_runner_waypoint.py', 'waypoint', include_c=False) # waypoint doesn't have S or C

