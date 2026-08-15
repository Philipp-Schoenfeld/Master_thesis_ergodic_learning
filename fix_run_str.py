import sys

with open('thesis_architecture/flow_matching_runner_ergodic.py', 'r') as f:
    content = f.read()

content = content.replace(
    'run_str = f"S{args.S}_nxi{args.nxi}_D{args.D}_flip{args.p_flip}"',
    'run_str = f"S{args.S}_nxi{args.nxi}_D{args.D}_C{args.copies_per_char}_flip{args.p_flip}"'
)
content = content.replace(
    'run_str  = f"ergodic_S{args.S}_nxi{args.nxi}_D{args.D}_flip{args.p_flip}"',
    'run_str  = f"ergodic_S{args.S}_nxi{args.nxi}_D{args.D}_C{args.copies_per_char}_flip{args.p_flip}"'
)

with open('thesis_architecture/flow_matching_runner_ergodic.py', 'w') as f:
    f.write(content)

