import numpy as np

entropies = [9.098, 9.224, 9.313]
names = ['II', 'H', 'N']

for n, e in zip(names, entropies):
    # Base calculation
    k = int(30 + (e - 9.0) * 100)
    # Clip between 20 and 80 (since T=100)
    k = np.clip(k, 20, 80)
    print(f"{n}: entropy={e:.3f} -> K={k}")
