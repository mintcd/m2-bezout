import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _compute_iteration_numba(history, n, Un, p1, p2, rand_vals, u, v, out):
    num_particles = Un.shape[0]
    for i in prange(num_particles):
        idx1 = Un[i]
        idx2 = n - 1 - idx1
        y1 = history[idx1, p1[i]]
        y2 = history[idx2, p2[i]]
        r = rand_vals[i]
        if r < u:
            out[i] = y1 + y2 + 1
        elif r < u + v:
            # max without calling Python's built-in
            out[i] = y1 if y1 >= y2 else y2
        else:
            out[i] = 0


def simulate_Yn_numba(u: float, v: float, N: int, num_particles: int):
    """Numba-accelerated simulator. Pre-generates random arrays per-iteration
    and uses a parallel njit kernel to compute the new particle values.
    """
    history = np.zeros((N + 1, num_particles), dtype=np.int64)
    out = np.empty(num_particles, dtype=np.int64)

    for nplusone in range(2, N + 1):
        n = nplusone - 1
        Un = np.random.randint(0, n, size=num_particles).astype(np.int64)
        p1 = np.random.randint(0, num_particles, size=num_particles).astype(np.int64)
        p2 = np.random.randint(0, num_particles, size=num_particles).astype(np.int64)
        rand_vals = np.random.rand(num_particles).astype(np.float64)

        _compute_iteration_numba(history, n, Un, p1, p2, rand_vals, u, v, out)
        history[n + 1, :] = out

    return history


def simulate_Yn_numpy(u: float, v: float, N: int, num_particles: int):
    """Reference pure-NumPy implementation (same logic as the notebook's
    `simulate_Yn`). Useful for micro-benchmarks.
    """
    history = np.zeros((N + 1, num_particles), dtype=np.int64)

    for nplusone in range(2, N + 1):
        n = nplusone - 1
        Un = np.random.randint(0, n, size=num_particles)
        idx1 = Un
        idx2 = n - 1 - Un
        p_idx1 = np.random.randint(0, num_particles, size=num_particles)
        p_idx2 = np.random.randint(0, num_particles, size=num_particles)
        Y1 = history[idx1, p_idx1]
        Y2 = history[idx2, p_idx2]
        rand_vals = np.random.rand(num_particles)
        mask_I = rand_vals < u
        mask_J = (rand_vals >= u) & (rand_vals < u + v)
        Y_new = np.zeros(num_particles, dtype=np.int64)
        if np.any(mask_I):
            Y_new[mask_I] = Y1[mask_I] + Y2[mask_I] + 1
        if np.any(mask_J):
            Y_new[mask_J] = np.maximum(Y1[mask_J], Y2[mask_J])
        history[n + 1] = Y_new

    return history


def benchmark_small(u: float = 0.5, v: float = 0.3, N: int = 40, num_particles: int = 2000):
    """Run a small benchmark: warm up the Numba kernel (compile), then time
    the pure-NumPy and Numba implementations and print results.
    """
    import time

    print("Warmup: compiling Numba kernel (this may take a moment)...")
    # small warmup to trigger compilation
    simulate_Yn_numba(u, v, 2, min(50, num_particles))

    print("Running pure-NumPy implementation...")
    t0 = time.time()
    simulate_Yn_numpy(u, v, N, num_particles)
    t_numpy = time.time() - t0
    print(f"NumPy version took {t_numpy:.3f} s")

    print("Running Numba implementation...")
    t0 = time.time()
    simulate_Yn_numba(u, v, N, num_particles)
    t_numba = time.time() - t0
    print(f"Numba version took {t_numba:.3f} s")

    if t_numba > 0:
        print(f"Speedup: {t_numpy / t_numba:.2f}x")
    else:
        print("Numba timing error (zero or near-zero runtime).")

    return t_numpy, t_numba


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Numba simulator and benchmark")
    parser.add_argument("--u", type=float, default=0.5, help="probability u")
    parser.add_argument("--v", type=float, default=0.3, help="probability v")
    parser.add_argument("--N", type=int, default=40, help="max N (inclusive)")
    parser.add_argument("--num_particles", type=int, default=2000, help="number of particles")
    parser.add_argument("--run-benchmark", dest="run_benchmark", action="store_true", help="Run the small compile+benchmark")
    args = parser.parse_args()

    if args.run_benchmark:
        benchmark_small(u=args.u, v=args.v, N=args.N, num_particles=args.num_particles)
    else:
        print("Running full Numba simulation (no timing)...")
        simulate_Yn_numba(args.u, args.v, args.N, args.num_particles)


if __name__ == "__main__":
    main()
