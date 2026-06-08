"""Optimized implementations of apply_T.

- `_apply_T_numba_core`: njit(parallel=True) per-particle kernel (recommended).
- `apply_T`: wrapper exposing `method='numba'|'vectorized'|'raw'`.

Concurrency notes:
- Writing to disjoint indices (mask1/mask2/mask3) is safe from a memory-race perspective,
  but using multiple Python threads that each call NumPy can cause contention/oversubscription
  (BLAS/OpenMP). The numba prange kernel avoids Python-level dispatch and is the recommended approach.
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _apply_T_numba_core(Z, u, v, alpha, rng_u, rand_vals, idx_j, idx_k, out):
    n = Z.shape[0]
    # probabilities computed once per-call
    c = (1.0 - u - v) / (u + v)
    p1 = (2.0 * c) / (1.0 + c)
    p2 = u * (1.0 - c)

    for i in prange(n):
        U = rng_u[i]
        r = rand_vals[i]
        z1 = Z[idx_j[i]]
        z2 = Z[idx_k[i]]
        U_alpha = U ** alpha
        comp = (1.0 - U) ** alpha

        if r < p1:
            out[i] = U_alpha * z1
        elif r < p1 + p2:
            out[i] = U_alpha * z1 + comp * z2
        else:
            v1 = U_alpha * z1
            v2 = comp * z2
            out[i] = v1 if v1 >= v2 else v2


def apply_T(Z, u, v, alpha, method='numba'):
    """Apply the operator T to empirical samples Z.

    Args:
      Z (1D array-like): empirical sample array.
      u, v, alpha (float): parameters.
      method (str): 'numba' (default) | 'vectorized' | 'raw'

    Returns:
      ndarray: transformed samples (dtype float64).

    Notes:
      - 'numba' uses a single parallel kernel (safe and scalable).
      - 'vectorized' uses NumPy operations and boolean masks (fewer Python-level loops,
        good when numba isn't available).
      - 'raw' reproduces the original masked-assignment code.
    """
    Z = np.asarray(Z)
    n = Z.shape[0]

    # pre-generate randomness and ancestor indices
    idx_j = np.random.randint(0, n, size=n).astype(np.int64)
    idx_k = np.random.randint(0, n, size=n).astype(np.int64)
    rng_u = np.random.rand(n).astype(np.float64)
    rand_vals = np.random.rand(n).astype(np.float64)

    if method == 'numba':
        out = np.empty(n, dtype=np.float64)
        Zf = Z.astype(np.float64)
        _apply_T_numba_core(Zf, u, v, alpha, rng_u, rand_vals, idx_j, idx_k, out)
        return out

    # shared quantities for non-numba paths
    Z1 = Z[idx_j].astype(np.float64)
    Z2 = Z[idx_k].astype(np.float64)
    U_alpha = np.power(rng_u, alpha)
    comp = np.power(1.0 - rng_u, alpha)
    t1 = U_alpha * Z1
    t2 = comp * Z2

    c = (1.0 - u - v) / (u + v)
    p1 = (2.0 * c) / (1.0 + c)
    p2 = u * (1.0 - c)

    mask1 = rand_vals < p1
    mask2 = (rand_vals >= p1) & (rand_vals < p1 + p2)
    mask3 = rand_vals >= p1 + p2

    out = np.empty(n, dtype=np.float64)

    if method == 'vectorized':
        # few temporaries, single-threaded NumPy operations
        out[mask1] = t1[mask1]
        out[mask2] = t1[mask2] + t2[mask2]
        out[mask3] = np.maximum(t1[mask3], t2[mask3])
        return out

    # method == 'raw': reproduce original masked-assignment semantics
    if method == 'raw':
        # same as 'vectorized' but mirrors original naming
        out[mask1] = U_alpha[mask1] * Z1[mask1]
        out[mask2] = (U_alpha[mask2] * Z1[mask2]) + (comp[mask2] * Z2[mask2])
        out[mask3] = np.maximum(U_alpha[mask3] * Z1[mask3], comp[mask3] * Z2[mask3])
        return out

    raise ValueError("Unknown method; choose 'numba', 'vectorized' or 'raw'.")
