import os
import re

with open('flow_matching_bspline_2d.py', 'r') as f:
    code = f.read()

# ==========================================
# Generate Chebyshev
# ==========================================
cheb_code = code

# 1. Remove bsplinax import
cheb_code = re.sub(r'# B-Spline library\n.*?\nfrom bsplinax\.bspline import BsplineBasisClamped\n', '', cheb_code, flags=re.DOTALL)

# 2. Replace basis generation
cheb_basis_code = """
# Precompute Chebyshev basis
t_vals = np.linspace(-1, 1, T)
B_np = np.zeros((T, NUM_COEFFS))
for k in range(NUM_COEFFS):
    B_np[:, k] = np.cos(k * np.arccos(t_vals))

B_mat = jnp.array(B_np) # Shape: (T, NUM_COEFFS)
B_outer = jnp.einsum('tk,tl->kl', B_mat, B_mat) * dt
G_R = jnp.kron(R_mat, B_outer)
"""
cheb_code = re.sub(r'# Precompute B-spline basis.*?G_R = jnp\.kron\(R_mat, B_outer\)', cheb_basis_code.strip(), cheb_code, flags=re.DOTALL)

# 3. String replacements
cheb_code = cheb_code.replace('2D Flow Matching Ergodic Coverage Testbench with B-Splines', '2D Flow Matching Ergodic Coverage Testbench with Chebyshev Polynomials')
cheb_code = cheb_code.replace('flow_matching_bspline_2d', 'flow_matching_chebyshev_2d')
cheb_code = cheb_code.replace('BSpline', 'Chebyshev')
cheb_code = cheb_code.replace('bspline', 'chebyshev')
cheb_code = cheb_code.replace('B-Spline', 'Chebyshev')
cheb_code = cheb_code.replace('B-spline', 'Chebyshev')
cheb_code = cheb_code.replace('NUM_CONTROL_POINTS', 'NUM_COEFFS')

with open('flow_matching_chebyshev_2d.py', 'w') as f:
    f.write(cheb_code)

# ==========================================
# Generate RBF
# ==========================================
rbf_code = code

# 1. Remove bsplinax import
rbf_code = re.sub(r'# B-Spline library\n.*?\nfrom bsplinax\.bspline import BsplineBasisClamped\n', '', rbf_code, flags=re.DOTALL)

# 2. Replace basis generation
rbf_basis_code = """
# Precompute RBF basis
t_vals = np.linspace(0, 1, T)
c_vals = np.linspace(0, 1, NUM_RBF_CENTERS)
sigma = 1.5 / NUM_RBF_CENTERS

B_np = np.zeros((T, NUM_RBF_CENTERS))
for k in range(NUM_RBF_CENTERS):
    B_np[:, k] = np.exp(-0.5 * ((t_vals - c_vals[k]) / sigma) ** 2)

B_mat = jnp.array(B_np) # Shape: (T, NUM_RBF_CENTERS)
B_outer = jnp.einsum('tk,tl->kl', B_mat, B_mat) * dt
G_R = jnp.kron(R_mat, B_outer)
"""
rbf_code = re.sub(r'# Precompute B-spline basis.*?G_R = jnp\.kron\(R_mat, B_outer\)', rbf_basis_code.strip(), rbf_code, flags=re.DOTALL)

# 3. String replacements
rbf_code = rbf_code.replace('2D Flow Matching Ergodic Coverage Testbench with B-Splines', '2D Flow Matching Ergodic Coverage Testbench with RBFs')
rbf_code = rbf_code.replace('flow_matching_bspline_2d', 'flow_matching_rbf_2d')
rbf_code = rbf_code.replace('BSpline', 'RBF')
rbf_code = rbf_code.replace('bspline', 'rbf')
rbf_code = rbf_code.replace('B-Spline', 'RBF')
rbf_code = rbf_code.replace('B-spline', 'RBF')
rbf_code = rbf_code.replace('NUM_CONTROL_POINTS', 'NUM_RBF_CENTERS')

with open('flow_matching_rbf_2d.py', 'w') as f:
    f.write(rbf_code)

print("Generated alternatives!")
