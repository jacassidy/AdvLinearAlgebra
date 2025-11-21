#!/usr/bin/env python3
"""
eigs_methods_QR_improved.py

Standalone script to experiment with QR-based eigenvalue computation
for the truncated matrix A_N from equation (4.8) in Chugunova & Pelinovsky.

It implements:

  - construct_A(N, eps)
  - gershgorin_max_radius(A)
  - extract_eigvals_from_quasi_triangular(H)
  - shifted_qr_eigs_naive(A, ...)    # eigenvalues = diag(H)  (WRONG for complex pairs)
  - shifted_qr_eigs_improved(A, ...) # eigenvalues = extracted from 1x1/2x2 blocks

Additionally, it can produce a λ-plane spectrum plot comparing
the naive and improved QR eigenvalues on the SAME axes.

Example:
    python eigs_methods_QR_improved.py --N 128 --eps 0.3 --verbose --check-eig \
        --plot-spectrum --spectrum-output qr_compare_128.png
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Matrix construction (equation 4.8)
# ---------------------------------------------------------------------------

def construct_A(N: int, eps: float) -> np.ndarray:
    """
    Construct the truncated N×N matrix A_N from (4.8).

    Entries for n = 1..N:

      A_{n,n}     = n
      A_{n,n+1}   = (eps/2) * n * (n+1)
      A_{n,n-1}   = -(eps/2) * n * (n-1)

    In 0-based Python indices:

      A[i, i]   = i+1
      A[i, i+1] = (eps/2) * (i+1)*(i+2)
      A[i, i-1] = -(eps/2) * (i+1)*i
    """
    A = np.zeros((N, N), dtype=np.float64)

    # Diagonal: n for n = 1..N
    n = np.arange(1, N + 1, dtype=float)
    A[np.arange(N), np.arange(N)] = n

    # Superdiagonal: (eps/2) * n * (n+1)
    i = np.arange(0, N - 1)
    A[i, i + 1] = 0.5 * eps * (i + 1) * (i + 2)

    # Subdiagonal: -(eps/2) * n * (n-1)
    i = np.arange(1, N)
    A[i, i - 1] = -0.5 * eps * (i + 1) * i

    return A


# ---------------------------------------------------------------------------
# Gershgorin diagnostics
# ---------------------------------------------------------------------------

def gershgorin_max_radius(A: np.ndarray) -> float:
    """
    Compute max Gershgorin radius R_i = sum_j |a_ij| - |a_ii|.
    """
    absA = np.abs(A)
    row_sums = absA.sum(axis=1)
    diag_abs = np.abs(np.diag(A))
    radii = row_sums - diag_abs
    return float(radii.max())


# ---------------------------------------------------------------------------
# Eigenvalue extraction from quasi-upper-triangular real Schur form
# ---------------------------------------------------------------------------

def extract_eigvals_from_quasi_triangular(H: np.ndarray, subdiag_tol: float = 1e-12):
    """
    Given a real/quasi-upper-triangular matrix H (real Schur form),
    extract its eigenvalues correctly:

      - 1x1 blocks -> real eigenvalue = H[i, i]
      - 2x2 blocks -> complex conjugate pair from the 2x2 block

    Parameters
    ----------
    H : (n, n) ndarray
        Quasi-upper-triangular matrix (output of real QR iteration).
        We assume H is real-valued.
    subdiag_tol : float
        Threshold below which H[i, i-1] is considered zero.

    Returns
    -------
    eigvals : (n,) complex ndarray
    """
    H = np.asarray(H)
    n = H.shape[0]
    eigvals = []
    i = n - 1

    while i >= 0:
        if i == 0 or abs(H[i, i - 1]) < subdiag_tol:
            # 1x1 block
            eigvals.append(H[i, i])
            i -= 1
        else:
            # 2x2 block represents a complex conjugate pair
            block = H[i - 1:i + 1, i - 1:i + 1]
            w = np.linalg.eigvals(block)  # complex conjugate pair
            eigvals.extend(w)
            i -= 2

    # Reverse to restore increasing index order (purely aesthetic)
    eigvals = eigvals[::-1]
    return np.array(eigvals, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Shifted QR (naive & improved)
# ---------------------------------------------------------------------------

def shifted_qr_eigs_naive(
    A: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 5000,
    verbose: bool = False,
    log_every: int = 25,
):
    """
    Shifted QR iteration, NAIVE eigenvalue extraction:

    - Performs Rayleigh-shifted QR on a real Hessenberg matrix.
    - At the end, returns eigvals = diag(H), which is only correct if H
      ends up fully upper-triangular (no 2x2 blocks). For general real
      matrices with complex eigenpairs, this *loses* imaginary parts.

    Parameters
    ----------
    A : ndarray
        Input matrix (N×N real).
    tol : float
        Target Gershgorin radius threshold.
    max_iter : int
        Maximum number of QR iterations.
    verbose : bool
        Print progress if True.
    log_every : int
        How often to print progress (if verbose).

    Returns
    -------
    eigvals : (N,) complex ndarray
        Naive eigenvalue approximation (real, in practice).
    H : (N, N) ndarray
        Final quasi-upper-triangular matrix.
    history : list of tuples
        (k, elapsed_time, offdiag_norm, gershgorin_radius)
    """
    H = A.astype(np.float64).copy()
    n = H.shape[0]
    I = np.eye(n, dtype=np.float64)

    history = []
    start = time.perf_counter()

    if verbose:
        print(
            f"[QR-naive] Starting shifted QR: n={n}, tol={tol:.1e}, max_iter={max_iter}"
        )

    for k in range(max_iter):
        mu = H[-1, -1]
        Q, R = np.linalg.qr(H - mu * I)
        H = R @ Q + mu * I

        off_norm = np.linalg.norm(np.tril(H, -1))
        g_max = gershgorin_max_radius(H)
        elapsed = time.perf_counter() - start
        history.append((k, elapsed, off_norm, g_max))

        should_log = (
            (k == 0)
            or (k % log_every == 0)
            or (g_max < 10 * tol)
            or (k == max_iter - 1)
        )

        if verbose and should_log:
            print(
                f"[QR-naive] iter={k:5d}, t={elapsed:8.3f}s, "
                f"||subdiag||={off_norm:.3e}, gersh_R={g_max:.3e}, target={tol:.1e}"
            )

        if g_max < tol:
            if verbose:
                print(
                    f"[QR-naive] Converged at iter={k}: "
                    f"Gershgorin radius {g_max:.3e} < tol={tol:.1e}"
                )
            break

    eigvals_real = np.diag(H).copy()  # this ignores 2x2 blocks
    eigvals = eigvals_real.astype(np.complex128)  # cast to complex for uniformity

    if verbose:
        total_time = time.perf_counter() - start
        print(
            f"[QR-naive] Finished. Total iters={len(history)}, total time={total_time:.3f}s, "
            f"final gersh_R={g_max:.3e}"
        )

    return eigvals, H, history


def shifted_qr_eigs_improved(
    A: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 5000,
    verbose: bool = False,
    log_every: int = 25,
    subdiag_tol: float = 1e-12,
):
    """
    Shifted QR iteration, IMPROVED eigenvalue extraction:

    - Same QR iteration as `shifted_qr_eigs_naive`, but at the end:
        * treats the final H as a real Schur (quasi-upper-triangular) form
        * uses 1x1 and 2x2 blocks to recover complex eigenvalues.

    Parameters
    ----------
    A : ndarray
        Input matrix (N×N real).
    tol : float
        Target Gershgorin radius threshold.
    max_iter : int
        Maximum number of QR iterations.
    verbose : bool
        Print progress if True.
    log_every : int
        How often to print progress (if verbose).
    subdiag_tol : float
        Threshold to decide if H[i, i-1] is "zero" when extracting blocks.

    Returns
    -------
    eigvals : (N,) complex ndarray
        Eigenvalue approximations (real and complex).
    H : (N, N) ndarray
        Final quasi-upper-triangular matrix.
    history : list of tuples
        (k, elapsed_time, offdiag_norm, gershgorin_radius)
    """
    H = A.astype(np.float64).copy()
    n = H.shape[0]
    I = np.eye(n, dtype=np.float64)

    history = []
    start = time.perf_counter()

    if verbose:
        print(
            f"[QR-improved] Starting shifted QR: n={n}, tol={tol:.1e}, max_iter={max_iter}"
        )

    for k in range(max_iter):
        mu = H[-1, -1]
        Q, R = np.linalg.qr(H - mu * I)
        H = R @ Q + mu * I

        off_norm = np.linalg.norm(np.tril(H, -1))
        g_max = gershgorin_max_radius(H)
        elapsed = time.perf_counter() - start
        history.append((k, elapsed, off_norm, g_max))

        should_log = (
            (k == 0)
            or (k % log_every == 0)
            or (g_max < 10 * tol)
            or (k == max_iter - 1)
        )

        if verbose and should_log:
            print(
                f"[QR-improved] iter={k:5d}, t={elapsed:8.3f}s, "
                f"||subdiag||={off_norm:.3e}, gersh_R={g_max:.3e}, target={tol:.1e}"
            )

        if g_max < tol:
            if verbose:
                print(
                    f"[QR-improved] Converged at iter={k}: "
                    f"Gershgorin radius {g_max:.3e} < tol={tol:.1e}"
                )
            break

    # Proper eigenvalue extraction from quasi-upper-triangular H
    eigvals = extract_eigvals_from_quasi_triangular(H, subdiag_tol=subdiag_tol)

    if verbose:
        total_time = time.perf_counter() - start
        print(
            f"[QR-improved] Finished. Total iters={len(history)}, total time={total_time:.3f}s, "
            f"final gersh_R={g_max:.3e}"
        )

    return eigvals, H, history


# ---------------------------------------------------------------------------
# Utilities for comparison / printing
# ---------------------------------------------------------------------------

def summarize_eigs(name: str, eigvals: np.ndarray, imag_tol: float = 1e-10):
    """
    Print a brief summary: min/max real/imag, number of non-real eigenvalues.
    """
    real = eigvals.real
    imag = eigvals.imag
    num_complex = int(np.sum(np.abs(imag) > imag_tol))
    print(f"=== {name} ===")
    print(f"Re range: [{real.min():.6e}, {real.max():.6e}]")
    print(f"Im range: [{imag.min():.6e}, {imag.max():.6e}]")
    print(f"# eigenvalues with |Im| > {imag_tol:.1e}: {num_complex}")
    print()


def compare_with_numpy(A: np.ndarray, eigvals_qr: np.ndarray, k_print: int = 5):
    """
    Compare improved QR eigenvalues to numpy.linalg.eig(A).

    For a fair comparison, we sort both sets of eigenvalues by real part
    then imaginary part and compute a max difference.
    """
    print("=== Comparison with numpy.linalg.eig ===")
    w, _ = np.linalg.eig(A)
    # Sort both
    def sorter(vals):
        return np.lexsort((vals.imag, vals.real))

    sort_qr = eigvals_qr[sorter(eigvals_qr)]
    sort_np = w[sorter(w)]

    if len(sort_qr) != len(sort_np):
        print(f"WARNING: size mismatch: QR has {len(sort_qr)}, numpy has {len(sort_np)}")

    m = min(len(sort_qr), len(sort_np))
    diff = sort_qr[:m] - sort_np[:m]
    max_err = np.max(np.abs(diff))
    print(f"Max |QR - numpy| over {m} eigenvalues: {max_err:.6e}")

    print(f"First {k_print} eigenvalues (numpy vs QR):")
    for i in range(min(k_print, m)):
        print(f"  i={i:3d}: numpy={sort_np[i]: .6e},  QR={sort_qr[i]: .6e}")
    print()


# ---------------------------------------------------------------------------
# Spectrum plotting: naive vs improved QR on same λ-plane axes
# ---------------------------------------------------------------------------

def plot_spectrum_compare(
    eig_naive: np.ndarray,
    eig_improved: np.ndarray,
    N: int,
    eps: float,
    output_file: str = "qr_compare.png",
    xlim=(-50, 50),
    ylim=(-25, 25),
):
    """
    Plot λ = i μ for both naive and improved QR eigenvalues
    on the same axis, with fixed limits and a square box.

    We also show both +λ and -λ to mimic the ± symmetry in the paper.

    - Naive QR: circles
    - Improved QR: crosses
    """
    lam_naive = 1j * eig_naive
    lam_improved = 1j * eig_improved

    # Reflect to show both +λ and -λ
    lam_naive_plot = np.concatenate([lam_naive, -lam_naive])
    lam_improved_plot = np.concatenate([lam_improved, -lam_improved])

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(lam_naive_plot.real, lam_naive_plot.imag,
               s=12, marker="o", alpha=0.7, label="QR naive (±λ)")
    ax.scatter(lam_improved_plot.real, lam_improved_plot.imag,
               s=20, marker="x", alpha=0.7, label="QR improved (±λ)")

    ax.axhline(0.0, linewidth=0.5)
    ax.axvline(0.0, linewidth=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"QR spectrum comparison (N={N}, eps={eps})")
    ax.legend()

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_box_aspect(1)

    print(f"[plot] Saving spectrum comparison to {output_file}")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare naive and improved shifted QR eigenvalue extraction "
                    "for the truncated matrix A_N and optionally plot the spectrum."
    )
    parser.add_argument("--N", type=int, default=128, help="Matrix size N.")
    parser.add_argument("--eps", type=float, default=0.3, help="Epsilon parameter.")
    parser.add_argument("--tol", type=float, default=1e-10, help="Gershgorin tolerance.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Max QR iterations.")
    parser.add_argument("--subdiag-tol", type=float, default=1e-12,
                        help="Subdiagonal tolerance for 1x1 vs 2x2 blocks.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed QR progress.")
    parser.add_argument("--check-eig", action="store_true",
                        help="Also compare improved QR to numpy.linalg.eig(A).")

    # new options to make the plot
    parser.add_argument("--plot-spectrum", action="store_true",
                        help="Generate λ-plane spectrum plot comparing naive vs improved QR.")
    parser.add_argument("--spectrum-output", type=str, default="qr_compare.png",
                        help="Output filename for spectrum comparison plot.")

    args = parser.parse_args()

    N = args.N
    eps = args.eps

    print(f"Building A for N={N}, eps={eps} ...")
    A = construct_A(N, eps)

    print("Running NAIVE shifted QR (diag(H) eigenvalues) ...")
    eig_naive, H_naive, hist_naive = shifted_qr_eigs_naive(
        A, tol=args.tol, max_iter=args.max_iter,
        verbose=args.verbose
    )
    summarize_eigs("QR-naive", eig_naive)

    print("Running IMPROVED shifted QR (block-based eigenvalues) ...")
    eig_improved, H_imp, hist_imp = shifted_qr_eigs_improved(
        A, tol=args.tol, max_iter=args.max_iter,
        verbose=args.verbose,
        subdiag_tol=args.subdiag_tol,
    )
    summarize_eigs("QR-improved", eig_improved)

    # Optional: compare improved QR to numpy.linalg.eig
    if args.check_eig:
        print("Computing numpy.linalg.eig(A) for comparison ...")
        compare_with_numpy(A, eig_improved, k_print=10)

    # Optional: plot spectrum comparison
    if args.plot_spectrum:
        plot_spectrum_compare(
            eig_naive=eig_naive,
            eig_improved=eig_improved,
            N=N,
            eps=eps,
            output_file=args.spectrum_output,
            xlim=(-50, 50),
            ylim=(-25, 25),
        )


if __name__ == "__main__":
    main()
