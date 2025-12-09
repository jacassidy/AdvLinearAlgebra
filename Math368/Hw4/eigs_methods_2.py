"""
eigs_methods.py

Numerical methods for eigenvalues/eigenvectors of the truncated matrix A_N
from equation (4.8) of Chugunova & Pelinovsky (2008).

Matrix A (size N×N) has entries (for n = 1..N):

  A_{n,n}     = n
  A_{n,n+1}   = (eps/2) * n * (n+1)
  A_{n,n-1}   = -(eps/2) * n * (n-1)

and satisfies A f_+ = -i λ f_+ (equation (4.7)), so eigenvalues in the λ–plane are
λ = i * μ where μ are eigenvalues of A.

This module provides:
  - construct_A(N, eps)
  - shifted_qr_eigs(A, ...)   # improved real QR with correct complex eigenvalues
  - eigenvectors_via_svd(A, eigvals)
  - arnoldi_iteration(A, m, v0)
  - arnoldi_eigs(A, num_eigs, m_subspace, which, tol, verbose)
"""

import time
import numpy as np


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
# Improved shifted QR on (real) Hessenberg matrix
# ---------------------------------------------------------------------------

def shifted_qr_eigs(
    A: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 5000,
    verbose: bool = False,
    log_every: int = 25,
    subdiag_tol: float = 1e-12,
):
    """
    Shifted QR iteration with correct complex eigenvalue extraction.

    - Uses a simple Rayleigh shift μ_k = H_{nn} at each step.
    - Works on a REAL matrix H and lets it converge to a real Schur form:
        * 1x1 blocks on the diagonal -> real eigenvalues
        * 2x2 blocks on the diagonal -> complex conjugate pairs
    - At the end, uses `extract_eigvals_from_quasi_triangular` to get
      all eigenvalues (real and complex).

    Stopping criterion: Gershgorin radius < tol or max_iter reached.

    Parameters
    ----------
    A : ndarray
        Input matrix (N×N, real).
    tol : float
        Target threshold on max Gershgorin radius.
    max_iter : int
        Maximum number of QR iterations.
    verbose : bool
        If True, prints progress.
    log_every : int
        Print every `log_every` iterations if verbose.
    subdiag_tol : float
        Threshold to decide if H[i, i-1] is "zero" when extracting blocks.

    Returns
    -------
    eigvals : (N,) complex ndarray
        Approximate eigenvalues μ of A.
    H : (N, N) real ndarray
        Final quasi-upper-triangular matrix.
    history : list of tuples
        (iteration, elapsed_time, offdiag_norm, gershgorin_radius)
    """
    H = A.astype(np.float64).copy()
    n = H.shape[0]
    I = np.eye(n, dtype=np.float64)

    history = []
    start = time.perf_counter()

    if verbose:
        print(
            f"[QR] Starting shifted QR (improved): n={n}, tol={tol:.1e}, "
            f"max_iter={max_iter}"
        )

    for k in range(max_iter):
        # Rayleigh shift from bottom-right entry
        mu = H[-1, -1]
        Q, R = np.linalg.qr(H - mu * I)
        H = R @ Q + mu * I

        # Diagnostics
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
                f"[QR] iter={k:5d}, t={elapsed:8.3f}s, "
                f"||subdiag||={off_norm:.3e}, "
                f"gersh_R={g_max:.3e}, target={tol:.1e}"
            )

        if g_max < tol:
            if verbose:
                print(
                    f"[QR] Converged at iter={k}: "
                    f"Gershgorin radius {g_max:.3e} < tol={tol:.1e}"
                )
            break

    # Proper eigenvalue extraction from quasi-upper-triangular H
    eigvals = extract_eigvals_from_quasi_triangular(H, subdiag_tol=subdiag_tol)

    if verbose:
        total_time = time.perf_counter() - start
        print(
            f"[QR] Finished. Total iters={len(history)}, total time={total_time:.3f}s, "
            f"final gersh_R={g_max:.3e}"
        )

    return eigvals, H, history


# ---------------------------------------------------------------------------
# Eigenvectors via least-squares / SVD (nullspace)
# ---------------------------------------------------------------------------

def eigenvectors_via_svd(A: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    """
    Compute eigenvectors by solving (A - μ I) v ≈ 0 in least-squares sense.

    For each eigenvalue μ, find the right singular vector of A - μ I
    corresponding to the smallest singular value.

    Parameters
    ----------
    A : (n, n) ndarray
        Real or complex; will be cast to complex.
    eigvals : (m,) ndarray
        Eigenvalues μ of A in some order.

    Returns
    -------
    V : (n, m) ndarray (complex)
        Column j is a unit eigenvector associated (approximately) with eigvals[j].
    """
    A_c = np.array(A, dtype=np.complex128)
    n = A_c.shape[0]
    m = len(eigvals)
    V = np.zeros((n, m), dtype=np.complex128)

    for j, lam in enumerate(eigvals):
        M = A_c - lam * np.eye(n, dtype=np.complex128)
        # SVD-based nullspace approximation
        _, _, vh = np.linalg.svd(M)
        v = vh[-1, :].conj().T
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        V[:, j] = v

    return V


# ---------------------------------------------------------------------------
# Arnoldi (Krylov)
# ---------------------------------------------------------------------------

def arnoldi_iteration(A: np.ndarray, m: int, v0=None):
    """
    Perform m-step Arnoldi iteration to build a Krylov subspace K_m(A, v0).

    Parameters
    ----------
    A : (n, n) ndarray (real or complex)
    m : int
        Dimension of the Krylov subspace (number of Arnoldi steps).
    v0 : (n,) array-like or None
        Initial vector (if None, a random complex vector is used).

    Returns
    -------
    Q : (n, m') ndarray
        Orthonormal basis for the Krylov subspace (m' ≤ m if happy breakdown).
    H : (m', m') ndarray
        Upper Hessenberg matrix representing A in this basis.
    """
    A_c = np.asarray(A, dtype=np.complex128)
    n = A_c.shape[0]

    if v0 is None:
        v = np.random.randn(n) + 1j * np.random.randn(n)
    else:
        v = np.array(v0, dtype=np.complex128)

    v = v / np.linalg.norm(v)

    Q = np.zeros((n, m), dtype=np.complex128)
    H = np.zeros((m, m), dtype=np.complex128)

    Q[:, 0] = v

    for j in range(m - 1):
        w = A_c @ Q[:, j]
        for i in range(j + 1):
            H[i, j] = np.vdot(Q[:, i], w)
            w = w - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] == 0:
            # happy breakdown: Krylov subspace is invariant
            return Q[:, : j + 1], H[: j + 1, : j + 1]
        Q[:, j + 1] = w / H[j + 1, j]

    # Last column of H
    w = A_c @ Q[:, m - 1]
    for i in range(m):
        H[i, m - 1] = np.vdot(Q[:, i], w)

    return Q, H


def arnoldi_eigs(
    A: np.ndarray,
    num_eigs: int | None = None,
    m_subspace: int | None = None,
    which: str = "LM",
    tol: float = 1e-8,
    verbose: bool = False,
):
    """
    Compute approximate eigenpairs using Arnoldi's method.

    Parameters
    ----------
    A : (n, n) ndarray (real or complex)
    num_eigs : int or None
        How many eigenvalues/eigenvectors to return. If None, return all
        eigenpairs of the projected matrix H.
    m_subspace : int or None
        Dimension of the Krylov subspace K_m(A, v0). If None, choose m = n
        if num_eigs is None, or m = min(n, max(2*num_eigs, 20)).
    which : {"LM", "SM"}
        Which eigenvalues of H to prefer:
          - "LM": largest magnitude
          - "SM": smallest magnitude
    tol : float
        Residual norm threshold used only for printing; we do NOT adaptively stop.
    verbose : bool
        If True, prints a summary with residual norms.

    Returns
    -------
    theta : (k,) ndarray (complex)
        Approximate eigenvalues (Ritz values) of A.
    U : (n, k) ndarray (complex)
        Corresponding Ritz vectors (approximate eigenvectors of A).
    info : dict
        Contains elapsed time and residual norms.
    """
    A_c = np.asarray(A, dtype=np.complex128)
    n = A_c.shape[0]

    if m_subspace is None:
        if num_eigs is None:
            m_subspace = n
        else:
            m_subspace = min(n, max(2 * num_eigs, 20))

    t0 = time.perf_counter()
    Q, H = arnoldi_iteration(A_c, m_subspace)
    elapsed = time.perf_counter() - t0

    # Eigen-decomposition of H
    w, Y = np.linalg.eig(H)  # H Y = Y diag(w)

    # Sort indices by magnitude
    if which.upper() == "SM":
        idx = np.argsort(np.abs(w))
    else:  # default: "LM"
        idx = np.argsort(-np.abs(w))

    if num_eigs is not None and num_eigs < len(idx):
        idx = idx[:num_eigs]

    theta = w[idx]
    Ysel = Y[:, idx]

    # Ritz vectors
    U = Q @ Ysel

    # Residual norms ||A u - theta u|| for each Ritz pair
    res = np.zeros_like(theta, dtype=float)
    for j, eigv in enumerate(theta):
        u = U[:, j]
        r = A_c @ u - eigv * u
        res[j] = np.linalg.norm(r)

    if verbose:
        print("[Arnoldi] elapsed = {:.4f} s".format(elapsed))
        for j, (lam, rnorm) in enumerate(zip(theta, res)):
            print(
                f"  j={j:3d}, theta={lam: .6e}, |theta|={abs(lam):.3e}, "
                f"residual={rnorm:.3e}"
            )

    info = {
        "elapsed": elapsed,
        "residuals": res,
        "theta_full": w,
    }

    return theta, U, info
