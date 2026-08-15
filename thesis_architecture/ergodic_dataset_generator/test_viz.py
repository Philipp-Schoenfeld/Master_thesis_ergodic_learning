from generate_dataset import plot_all_targets
from shape_library import train_shape_names, VALIDATION_SHAPES
import os

all_names = VALIDATION_SHAPES + train_shape_names(500)
save_dir = os.path.join('visualizations', 'all_targets')
print(f"Generating targets for {len(all_names)} shapes...")
plot_all_targets(all_names, save_dir)
