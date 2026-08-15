import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/philipp/Documents/Uni/Master_thesis/thesis_architecture/ergodic_dataset_generator')
from shape_library import get_shape

shape_def = get_shape('test_gmm_2')
means = np.array(shape_def['means'])
covs = np.array(shape_def['covs'])
weights = np.array(shape_def['weights'])
weights = weights / np.sum(weights)

# TSP
unvisited = list(range(len(means)))
curr_idx = 0 # start at 0
path_indices = [curr_idx]
unvisited.remove(curr_idx)
while unvisited:
    curr_mean = means[curr_idx]
    dists = [np.linalg.norm(curr_mean - means[idx]) for idx in unvisited]
    best = np.argmin(dists)
    curr_idx = unvisited.pop(best)
    path_indices.append(curr_idx)

# Build continuous path
points = []
for idx in path_indices:
    mu = means[idx]
    cov = covs[idx]
    w = weights[idx]
    
    # eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    # vals are variances. stddev is sqrt(vals)
    E1 = vecs[:, 1] * np.sqrt(vals[1])
    E2 = vecs[:, 0] * np.sqrt(vals[0])
    
    # Lissajous
    # Number of points proportional to weight (min 10)
    n_pts = max(10, int(w * 500))
    tau = np.linspace(0, 2*np.pi, n_pts)
    
    # p(tau) = mu + 2*E1*sin(tau) + 2*E2*sin(2*tau)
    # wait, sin(tau) and sin(2*tau) is a figure 8.
    # To cover more, sin(3*tau) and sin(2*tau)
    curve = mu[None, :] + 2.0 * np.outer(np.sin(3*tau), E1) + 2.0 * np.outer(np.sin(2*tau), E2)
    
    # add transit from previous point
    if len(points) > 0:
        last_pt = points[-1][-1]
        transit = np.linspace(last_pt, curve[0], 10)[1:-1]
        points.append(transit)
        
    points.append(curve)

points = np.vstack(points)

# Interpolate to exactly 200 points based on distance (or just uniform if we want to preserve weight distribution)
# Wait, if we just uniformly subsample `points` to 200 points, it automatically spends more time in high-weight components because we allocated `n_pts` proportionally!
idx_eval = np.linspace(0, len(points)-1, 201)
p_traj = np.column_stack([
    np.interp(idx_eval, np.arange(len(points)), points[:, 0]),
    np.interp(idx_eval, np.arange(len(points)), points[:, 1])
])

from scipy.ndimage import gaussian_filter1d
p_traj = gaussian_filter1d(p_traj, sigma=2, axis=0, mode='nearest')

plt.plot(p_traj[:, 0], p_traj[:, 1], 'w.-')
plt.scatter(means[:, 0], means[:, 1], c='r')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('test_lissajous.png')
print("Saved test_lissajous.png with shape", p_traj.shape)
