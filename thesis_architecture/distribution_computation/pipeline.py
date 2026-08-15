import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path so we can import shape_library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ergodic_dataset_generator.shape_library import get_shape, pdf_on_grid

# Import all our new encoding modules
import encode_gmm
import encode_spectral
import encode_grid
import encode_particles
import encode_sdf
import encode_hybrid

# 1. Select 10 specific shapes to cover the complexity spectrum
# 3 Letters, 3 Random GMMs, 4 Analytical Polygons
TARGET_SHAPES = [
    'A', 'greek_upper_23', 'korean_15',
    'rand_gmm_42', 'rand_gmm_7', 'rand_gmm_100',
    'rand_ana_poly_5', 'rand_ana_poly_22', 'rand_ana_poly_99', 'rand_ana_poly_150'
]

def main():
    print("Generating high-res Original Density Maps (256x256)...")
    original_maps = []
    
    for name in TARGET_SHAPES:
        shape_def = get_shape(name)
        # Note: pdf_on_grid returns (density, gx, gy)
        d_map, _, _ = pdf_on_grid(shape_def, resolution=256)
        
        # Normalize to [0, 1] for visual consistency
        if d_map.max() > 0:
            d_map /= d_map.max()
            
        original_maps.append(d_map)
        print(f" Loaded {name}")

    methods = {
        'GMM': encode_gmm,
        'Spectral': encode_spectral,
        'Grid': encode_grid,
        'Particles': encode_particles,
        'SDF_Contours': encode_sdf,
        'Hybrid': encode_hybrid
    }
    
    os.makedirs('results', exist_ok=True)

    for method_name, module in methods.items():
        print(f"\nProcessing Pipeline: {method_name}")
        
        fig, axes = plt.subplots(3, 10, figsize=(25, 8))
        fig.suptitle(f"Distribution Encoding Analysis: {method_name}", fontsize=20, y=0.98)
        
        for i, (name, d_map) in enumerate(zip(TARGET_SHAPES, original_maps)):
            # Row 0: Original
            ax_orig = axes[0, i]
            ax_orig.imshow(d_map, origin='lower', extent=[0, 1, 0, 1], cmap='Blues')
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            if i == 0:
                ax_orig.set_ylabel("Original\n(256x256)", fontsize=12, fontweight='bold')
            ax_orig.set_title(name, fontsize=10)
            
            # Encode
            print(f"  Encoding {name}...")
            encoded_data = module.encode(d_map)
            
            # Row 1: Encoded Viz
            ax_enc = axes[1, i]
            module.viz_encoded(ax_enc, encoded_data)
            if i == 0:
                ax_enc.set_ylabel(f"Encoded", fontsize=12, fontweight='bold')
                
            # Revert
            print(f"  Reverting {name}...")
            reverted_map = module.revert(encoded_data, resolution=256)
            
            # Row 2: Reverted Viz
            ax_rev = axes[2, i]
            ax_rev.imshow(reverted_map, origin='lower', extent=[0, 1, 0, 1], cmap='Blues')
            ax_rev.set_xticks([])
            ax_rev.set_yticks([])
            if i == 0:
                ax_rev.set_ylabel("Reverted\n(256x256)", fontsize=12, fontweight='bold')
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = f"results/pipeline_{method_name}.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
