import numpy as np
import matplotlib.pyplot as plt

def generate_organic(seed, pts=200):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, pts)
    kind = rng.choice(['amoeba', 'cloud', 'leaf', 'starfish'])
    
    if kind == 'amoeba':
        r = 0.3
        for _ in range(rng.integers(3, 7)):
            k = rng.integers(2, 6)
            phase = rng.uniform(0, 2*np.pi)
            amp = rng.uniform(0.02, 0.08)
            r += amp * np.sin(k * t + phase)
        x = 0.5 + r * np.cos(t)
        y = 0.5 + r * np.sin(t)
    elif kind == 'cloud':
        r = 0.25
        for _ in range(rng.integers(4, 9)):
            k = rng.integers(3, 9)
            phase = rng.uniform(0, 2*np.pi)
            amp = rng.uniform(0.03, 0.07)
            # using abs(sin) for puffy cloud edges
            r += amp * np.abs(np.sin(k * t / 2 + phase))
        x = 0.5 + r * np.cos(t)
        y = 0.5 + r * np.sin(t)
    elif kind == 'leaf':
        # r = a * (1 + sin(t)) * (1 + c * cos(8t))
        t_leaf = t - np.pi/2 # point upwards
        r = 0.15 * (1 + np.sin(t_leaf)) * (1 + 0.3 * np.cos(rng.integers(4, 10) * t_leaf))
        # add noise
        r += rng.uniform(0, 0.02, pts)
        x = 0.5 + r * np.cos(t)
        y = 0.5 + r * np.sin(t)
    elif kind == 'starfish':
        k = rng.integers(4, 7)
        r = 0.3 + 0.15 * np.cos(k * t) + 0.05 * np.cos(k * 2 * t)
        x = 0.5 + r * np.cos(t)
        y = 0.5 + r * np.sin(t)
        
    return x, y

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    x, y = generate_organic(i + 10)
    axes[i].plot(x, y)
    axes[i].set_aspect('equal')
plt.savefig('test_organic.png')
