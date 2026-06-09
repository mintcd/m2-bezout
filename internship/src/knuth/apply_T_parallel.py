"""Optimized implementations of `apply_T`.

Two main helpers:
- `_apply_T_kernel_numba`: a Numba-parallel per-particle kernel.
- `apply_T_numba`: Python wrapper that generates randomness and calls the kernel.

Usage:
from src.knuth.apply_T_parallel import apply_T_numba
Z_new = apply_T_numba(Z, u, v, alpha)
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _apply_T_kernel_numba(Z, U, rand_vals, idx_j, idx_k, u, v, alpha):
    n = Z.shape[0]
    out = np.empty(n, dtype=np.float64)

    # branch probabilities (kept same as the original implementation)
    c = (1.0 - u - v) / (u + v)
    p1 = (2.0 * c) / (1.0 + c)
    p2 = u * (1.0 - c)
    p12 = p1 + p2

    for i in prange(n):
        j = idx_j[i]
        k = idx_k[i]
        z1 = Z[j]
        z2 = Z[k]

        ua = U[i] ** alpha
        ca = (1.0 - U[i]) ** alpha
        r = rand_vals[i]

        if r < p1:
            out[i] = ua * z1
        elif r < p12:
            out[i] = ua * z1 + ca * z2
        else:
            t1 = ua * z1
            t2 = ca * z2
            out[i] = t1 if t1 >= t2 else t2

    return out


def apply_T_numba(Z, u, v, alpha):
    """Parallel apply_T implementation using Numba.

    This wrapper pre-generates the random numbers/indices in NumPy
    and then dispatches the per-particle work to a Numba njit(parallel=True)
    kernel. This avoids large temporary boolean-indexed arrays and
    parallelizes the hot loop in C.

    Args:
      Z: 1D numpy array-like (will be cast to float64)
      u, v, alpha: floats

    Returns:
      Z_new: 1D numpy array (float64)
    """
    n = int(np.asarray(Z).shape[0])
    if n == 0:
        return np.array([], dtype=np.float64)

    Zf = np.asarray(Z, dtype=np.float64)

    # Pre-generate randomness with NumPy (fast and reproducible if you set RNG state)
    idx_j = np.random.randint(0, n, size=n).astype(np.int64)
    idx_k = np.random.randint(0, n, size=n).astype(np.int64)
    U = np.random.rand(n).astype(np.float64)
    rand_vals = np.random.rand(n).astype(np.float64)

    return _apply_T_kernel_numba(Zf, U, rand_vals, idx_j, idx_k, u, v, alpha)
