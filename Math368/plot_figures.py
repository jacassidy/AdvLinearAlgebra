#!/usr/bin/env python3
"""
plot_figures.py

Generate numerical analogues of Figs. 4 and 5 from Chugunova & Pelinovsky (2008)
using the truncated matrix A_N from (4.8) and TWO numerical methods:

  - QR-based method (shifted QR on Hessenberg)
  - Arnoldi/Krylov method

Each "left" / "right" figure is a single panel:
  - Fig 4-left:  N = 128  (spectrum)
  - Fig 4-right: N = 1024 (spectrum)
  - Fig 5-left:  cos(f_n, f_{n+1}) for n ~ 1..20
  - Fig 5-right: cos(f_1, f_2) vs epsilon

Axes are fixed as requested:
  - Fig 4: Re(λ) in [-50, 50], Im(λ) in [-25, 25]
  - Fig 5-left: n in [0, 20], cos in [0, 1]
  - Fig 5-right: eps in [0, 2], cos in [0, 1]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from eigs_methods_2 import (
    construct_A,
    shifted_qr_eigs,
    eigenvectors_via_svd,
    arnoldi_eigs,
)


# ---------------------------------------------------------------------------
# Utility: cosine of angle between two eigenvectors
# ---------------------------------------------------------------------------

def cos_angle(v: np.ndarray, w: np.ndarray) -> float:
    """Compute cos( f̂, ĝ ) = |(f, g)| / (||f|| * ||g||)."""
    num = np.vdot(v, w)  # conjugate(v) · w
    denom = np.linalg.norm(v) * np.linalg.norm(w)
    if denom == 0:
        return 0.0
    return float(np.abs(num / denom))


# ---------------------------------------------------------------------------
# Figure 4: single-panel spectrum plot (QR + Arnoldi)
# ---------------------------------------------------------------------------

def compute_spectrum_lambda_qr(
    A: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 5000,
    verbose: bool = False,
):
    """Eigenvalues via shifted QR; returns λ = i μ and QR history."""
    if verbose:
        print("[Fig4][QR] Running shifted QR ...")
    mu_qr, _, hist_qr = shifted_qr_eigs(A, tol=tol, max_iter=max_iter, verbose=verbose)
    lam_qr = 1j * mu_qr
    return lam_qr, hist_qr


def compute_spectrum_lambda_arnoldi(
    A: np.ndarray,
    verbose: bool = False,
):
    """
    Eigenvalues via Arnoldi; returns λ = i μ and info dict.

    We take m_subspace = n and num_eigs=None so Arnoldi works effectively
    on the full space (OK for the N we're using here).
    """
    n = A.shape[0]
    if verbose:
        print(f"[Fig4][Arnoldi] Running Arnoldi on n={n} (full subspace) ...")
    theta, U, info = arnoldi_eigs(
        A, num_eigs=None, m_subspace=n, which="LM", tol=1e-10, verbose=verbose
    )
    lam_arn = 1j * theta
    return lam_arn, info


def plot_figure4_single(
    N: int,
    eps: float,
    output_file: str,
    verbose: bool = False,
):
    """
    Single-panel Fig. 4-style spectrum plot for given N and eps.

    Plots BOTH:
      - QR eigenvalues as circles
      - Arnoldi eigenvalues as Xs

    We also reflect to show both +λ and -λ (upper and lower half-plane),
    to mimic the symmetry of the full operator in the paper.

    Axis limits are FIXED to:
      - Re(λ) in [-50, 50]
      - Im(λ) in [-25, 25]
    """
    if verbose:
        print(f"[Fig4] Single panel: N={N}, eps={eps}")
        print("[Fig4] Building A ...")

    A = construct_A(N, eps)

    # --- QR eigenvalues ---
    lam_qr, hist_qr = compute_spectrum_lambda_qr(A, verbose=verbose)
    if verbose:
        last_iter, last_time, _, last_g = hist_qr[-1]
        print(
            f"[Fig4][QR] done: iters={last_iter}, time={last_time:.3f}s, "
            f"final gersh_R={last_g:.3e}"
        )

    # --- Arnoldi eigenvalues ---
    lam_arn, info_arn = compute_spectrum_lambda_arnoldi(A, verbose=verbose)

    # --- Reflect to simulate full ±λ symmetry ---
    lam_qr_plot = np.concatenate([lam_qr, -lam_qr])
    lam_arn_plot = np.concatenate([lam_arn, -lam_arn])

    # Optional debug prints
    if verbose:
        def debug_lam(name, lam):
            imag = lam.imag
            real = lam.real
            print(
                f"[Fig4][debug] {name}: "
                f"Re(λ) in [{real.min():.3e}, {real.max():.3e}], "
                f"Im(λ) in [{imag.min():.3e}, {imag.max():.3e}]"
            )
            print(
                f"    counts: Im<0: {(imag < 0).sum()}, "
                f"Im=0: {(imag == 0).sum()}, Im>0: {(imag > 0).sum()}"
            )

        debug_lam("QR (±)", lam_qr_plot)
        debug_lam("Arnoldi (±)", lam_arn_plot)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(lam_qr_plot.real, lam_qr_plot.imag,
               s=12, marker="o", alpha=0.7, label="QR")
    ax.scatter(lam_arn_plot.real, lam_arn_plot.imag,
               s=20, marker="x", alpha=0.7, label="Arnoldi")

    ax.axhline(0.0, linewidth=0.5)
    ax.axvline(0.0, linewidth=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"eps = {eps}, N = {N}")
    ax.legend()

    # FIXED limits as requested
    ax.set_xlim(-50, 50)
    ax.set_ylim(-25, 25)

    # Make the plot box square (even though ranges differ)
    ax.set_box_aspect(1)

    fig.suptitle("Spectrum of truncated difference problem in λ-plane")

    if verbose:
        print(f"[Fig4] Saving figure to {output_file}")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Helper for Fig. 5: pick eigenvectors near imaginary axis
# ---------------------------------------------------------------------------

def _select_axis_family(lam: np.ndarray,
                        V: np.ndarray,
                        num_modes: int,
                        imag_axis_tol: float,
                        verbose: bool,
                        label: str):
    """
    Given λ, eigenvectors V, pick a "family" concentrated near the imaginary
    axis and return the first `num_modes` eigenvectors in that family.
    """
    # Order eigenvalues by closeness to imaginary axis
    idx_sorted = np.argsort(np.abs(lam.real) + 1e-12 * np.abs(lam.imag))
    lam_sorted = lam[idx_sorted]
    V_sorted = V[:, idx_sorted]

    # Filter for nearly purely imaginary
    mask = np.abs(lam_sorted.real) < imag_axis_tol
    lam_axis = lam_sorted[mask]
    V_axis = V_sorted[:, mask]

    if len(lam_axis) < num_modes + 1:
        if verbose:
            print(
                f"[{label}] Only {len(lam_axis)} eigenvalues within imag_axis_tol={imag_axis_tol}, "
                "falling back to first ones."
            )
        lam_axis = lam_sorted[: num_modes + 1]
        V_axis = V_sorted[:, : num_modes + 1]

    V_use = V_axis[:, : num_modes]
    return V_use


# ---------------------------------------------------------------------------
# Figure 5 left: cos(f_n, f_{n+1}) (QR + Arnoldi)
# ---------------------------------------------------------------------------

def figure5_left_cos_angles(
    N: int,
    eps: float,
    num_modes: int = 20,
    qr_tol: float = 1e-10,
    qr_max_iter: int = 5000,
    imag_axis_tol: float = 1e-4,
    output_file: str = "fig5_left.png",
    verbose: bool = False,
):
    """
    Fig. 5 (left) analogue.

    For BOTH methods:
      - build A_N
      - compute eigenvalues (QR / Arnoldi)
      - compute eigenvectors
      - convert to λ = i μ
      - pick eigenvalues closest to imaginary axis
      - compute cos( f_n, f_{n+1} ) for first num_modes

    Then plot both curves on the same axes.

    Axis limits:
      - x: n in [0, 20]
      - y: cos in [0, 1]
    """
    if verbose:
        print(f"[Fig5-left] N={N}, eps={eps}")
        print("[Fig5-left] Building A ...")
    A = construct_A(N, eps)

    # ----- QR -----
    if verbose:
        print("[Fig5-left][QR] Running shifted QR ...")
    mu_qr, _, hist_qr = shifted_qr_eigs(
        A, tol=qr_tol, max_iter=qr_max_iter, verbose=verbose
    )
    if verbose:
        last_iter, last_time, _, last_g = hist_qr[-1]
        print(
            f"[Fig5-left][QR] done: iters={last_iter}, time={last_time:.3f}s, "
            f"final gersh_R={last_g:.3e}"
        )

    if verbose:
        print("[Fig5-left][QR] Computing eigenvectors via SVD ...")
    V_qr = eigenvectors_via_svd(A, mu_qr)
    lam_qr = 1j * mu_qr

    V_qr_use = _select_axis_family(
        lam_qr, V_qr, num_modes=num_modes,
        imag_axis_tol=imag_axis_tol,
        verbose=verbose,
        label="Fig5-left[QR]"
    )

    cos_qr = []
    for k in range(num_modes - 1):
        f_n = V_qr_use[:, k]
        f_next = V_qr_use[:, k + 1]
        c = cos_angle(f_n, f_next)
        cos_qr.append(c)
        if verbose and (k < 3 or k == num_modes - 2):
            print(f"[Fig5-left][QR] n={k+1:3d}, cos={c:.6f}")

    # ----- Arnoldi -----
    if verbose:
        print("[Fig5-left][Arnoldi] Running Arnoldi ...")
    theta_arn, U_arn, info_arn = arnoldi_eigs(
        A, num_eigs=None, m_subspace=N, which="LM", tol=1e-10, verbose=verbose
    )
    lam_arn = 1j * theta_arn

    V_arn_use = _select_axis_family(
        lam_arn, U_arn, num_modes=num_modes,
        imag_axis_tol=imag_axis_tol,
        verbose=verbose,
        label="Fig5-left[Arnoldi]"
    )

    cos_arn = []
    for k in range(num_modes - 1):
        f_n = V_arn_use[:, k]
        f_next = V_arn_use[:, k + 1]
        c = cos_angle(f_n, f_next)
        cos_arn.append(c)
        if verbose and (k < 3 or k == num_modes - 2):
            print(f"[Fig5-left][Arnoldi] n={k+1:3d}, cos={c:.6f}")

    # ----- Plot -----
    indices = np.arange(1, num_modes)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(indices, cos_qr, marker="o", label="QR")
    ax.plot(indices, cos_arn, marker="x", label="Arnoldi")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\cos(\widehat{f_n, f_{n+1}})$")
    ax.grid(True)
    ax.set_title(f"eps = {eps}, N = {N}")
    ax.legend()

    # FIXED axes: n 0..20, cos 0..1
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)

    if verbose:
        print(f"[Fig5-left] Saving figure to {output_file}")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 right: cos(f1, f2) vs eps (QR + Arnoldi)
# ---------------------------------------------------------------------------

def figure5_right_cos_vs_eps(
    N: int,
    eps_min: float = 0.0,
    eps_max: float = 2.0,
    num_eps: int = 20,
    qr_tol: float = 1e-10,
    qr_max_iter: int = 5000,
    imag_axis_tol: float = 1e-4,
    output_file: str = "fig5_right.png",
    verbose: bool = False,
):
    """
    Fig. 5 (right) analogue:

      For each eps in [eps_min, eps_max], for BOTH methods:
        - build A_N
        - compute eigenpairs
        - select the two eigenvalues/eigenvectors closest to the imaginary axis
        - compute cos(f1, f2)

      Then plot two curves:
        cos_QR(eps), cos_Arnoldi(eps).

    Axis limits:
      - x: eps in [0, 2]
      - y: cos in [0, 1]
    """
    eps_values = np.linspace(eps_min, eps_max, num_eps)
    cos_qr = []
    cos_arn = []

    if verbose:
        print(
            f"[Fig5-right] Starting eps sweep: N={N}, "
            f"eps in [{eps_min}, {eps_max}] with {num_eps} points."
        )

    for idx, eps in enumerate(eps_values):
        if verbose:
            print(
                f"[Fig5-right] ({idx+1}/{num_eps}) eps={eps:.4f}: "
                "building A ..."
            )
        A = construct_A(N, eps)

        # --- QR ---
        if verbose:
            print("    [QR] running shifted QR ...")
        mu_qr, _, hist_qr = shifted_qr_eigs(
            A, tol=qr_tol, max_iter=qr_max_iter, verbose=False
        )
        if verbose:
            last_iter, last_time, _, last_g = hist_qr[-1]
            print(
                f"    [QR] QR done: iters={last_iter}, time={last_time:.3f}s, "
                f"final gersh_R={last_g:.3e}"
            )
            print("    [QR] computing eigenvectors via SVD ...")

        V_qr = eigenvectors_via_svd(A, mu_qr)
        lam_qr = 1j * mu_qr

        idx_sorted_qr = np.argsort(np.abs(lam_qr.real) + 1e-12 * np.abs(lam_qr.imag))
        lam_qr_sorted = lam_qr[idx_sorted_qr]
        V_qr_sorted = V_qr[:, idx_sorted_qr]

        mask_qr = np.abs(lam_qr_sorted.real) < imag_axis_tol
        cand_qr = np.nonzero(mask_qr)[0]
        if len(cand_qr) >= 2:
            i1_qr, i2_qr = cand_qr[:2]
        else:
            i1_qr, i2_qr = 0, 1

        f1_qr = V_qr_sorted[:, i1_qr]
        f2_qr = V_qr_sorted[:, i2_qr]
        c_qr = cos_angle(f1_qr, f2_qr)
        cos_qr.append(c_qr)

        if verbose:
            print(
                f"    [QR] cos(f1, f2)={c_qr:.6f} (indices {i1_qr}, {i2_qr})"
            )

        # --- Arnoldi ---
        if verbose:
            print("    [Arnoldi] running Arnoldi ...")
        theta_arn, U_arn, info_arn = arnoldi_eigs(
            A, num_eigs=None, m_subspace=N, which="LM", tol=1e-10, verbose=False
        )
        lam_arn = 1j * theta_arn

        idx_sorted_arn = np.argsort(np.abs(lam_arn.real) + 1e-12 * np.abs(lam_arn.imag))
        lam_arn_sorted = lam_arn[idx_sorted_arn]
        U_arn_sorted = U_arn[:, idx_sorted_arn]

        mask_arn = np.abs(lam_arn_sorted.real) < imag_axis_tol
        cand_arn = np.nonzero(mask_arn)[0]
        if len(cand_arn) >= 2:
            i1_arn, i2_arn = cand_arn[:2]
        else:
            i1_arn, i2_arn = 0, 1

        f1_arn = U_arn_sorted[:, i1_arn]
        f2_arn = U_arn_sorted[:, i2_arn]
        c_arn = cos_angle(f1_arn, f2_arn)
        cos_arn.append(c_arn)

        if verbose:
            print(
                f"    [Arnoldi] cos(f1, f2)={c_arn:.6f} (indices {i1_arn}, {i2_arn})"
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps_values, cos_qr, marker="o", label="QR")
    ax.plot(eps_values, cos_arn, marker="x", label="Arnoldi")
    ax.set_xlabel("eps")
    ax.set_ylabel(r"$\cos(\widehat{f_1, f_2})$")
    ax.grid(True)
    ax.set_title(f"cos(angle) between first two eigenvectors vs eps (N = {N})")
    ax.legend()

    # FIXED axes: eps 0..2, cos 0..1
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)

    if verbose:
        print(f"[Fig5-right] Saving figure to {output_file}")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate numerical analogues of Figs. 4 and 5 with QR and Arnoldi."
    )
    parser.add_argument(
        "--figure",
        type=str,
        required=True,
        choices=["4-left", "4-right", "5-left", "5-right", "all"],
        help="Which figure to generate.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Matrix size N (overrides defaults for some figures).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Epsilon value for Fig. 4 or Fig. 5-left (overrides defaults).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (PNG). If omitted, a default name is used.",
    )
    parser.add_argument(
        "--eps-min",
        type=float,
        default=0.0,
        help="Minimum epsilon for Fig. 5-right.",
    )
    parser.add_argument(
        "--eps-max",
        type=float,
        default=2.0,
        help="Maximum epsilon for Fig. 5-right.",
    )
    parser.add_argument(
        "--num-eps",
        type=int,
        default=20,
        help="Number of epsilon samples for Fig. 5-right.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )

    args = parser.parse_args()
    fig_name = args.figure
    verbose = args.verbose

    if fig_name == "4-left":
        # default: N=128, eps=0.3
        N = args.N if args.N is not None else 128
        eps = args.epsilon if args.epsilon is not None else 0.3
        out = args.output if args.output is not None else "fig4_left.png"
        plot_figure4_single(N=N, eps=eps, output_file=out, verbose=verbose)

    elif fig_name == "4-right":
        # default: N=1024, eps=0.3
        N = args.N if args.N is not None else 1024
        eps = args.epsilon if args.epsilon is not None else 0.3
        out = args.output if args.output is not None else "fig4_right.png"
        plot_figure4_single(N=N, eps=eps, output_file=out, verbose=verbose)

    elif fig_name == "5-left":
        # default: N=1024, eps=0.1
        N = args.N if args.N is not None else 1024
        eps = args.epsilon if args.epsilon is not None else 0.1
        out = args.output if args.output is not None else "fig5_left.png"
        figure5_left_cos_angles(
            N=N, eps=eps, num_modes=20, output_file=out, verbose=verbose
        )

    elif fig_name == "5-right":
        # default: N=512
        N = args.N if args.N is not None else 512
        out = args.output if args.output is not None else "fig5_right.png"
        figure5_right_cos_vs_eps(
            N=N,
            eps_min=args.eps_min,
            eps_max=args.eps_max,
            num_eps=args.num_eps,
            output_file=out,
            verbose=verbose,
        )

    elif fig_name == "all":
        # "all" uses defaults and verbose so you see where it is
        plot_figure4_single(
            N=128, eps=0.3, output_file="fig4_left.png", verbose=True
        )
        plot_figure4_single(
            N=1024, eps=0.3, output_file="fig4_right.png", verbose=True
        )
        figure5_left_cos_angles(
            N=1024, eps=0.1, num_modes=20, output_file="fig5_left.png", verbose=True
        )
        figure5_right_cos_vs_eps(
            N=512,
            eps_min=0.0,
            eps_max=2.0,
            num_eps=20,
            output_file="fig5_right.png",
            verbose=True,
        )

    else:
        raise ValueError(f"Unknown figure name: {fig_name}")


if __name__ == "__main__":
    main()
