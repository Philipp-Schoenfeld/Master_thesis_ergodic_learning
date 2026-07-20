import numpy as np
import scipy.linalg

def gram_schmidt_symm(A):
    """
    Symmetric Löwdin orthogonalization.
    A is (N, M), where columns are basis vectors.
    Returns an orthogonalized matrix of the same shape.
    """
    S = A.T @ A
    S_inv_half = scipy.linalg.fractional_matrix_power(S, -0.5)
    return A @ S_inv_half
