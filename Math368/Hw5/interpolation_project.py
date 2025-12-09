#!/usr/bin/env python3
"""
Interpolation experiments for:
- Reproducing Fig. 2 and Fig. 3 from the paper (conceptually)
- Running two extra functions with similar regularity properties

Usage:
    python interpolation_project.py fig2
    python interpolation_project.py fig3
    python interpolation_project.py extra
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Basic node generators
# -----------------------------

def equally_spaced_nodes(a, b, n):
    """
    Return n+1 equally spaced nodes in [a, b].
    """
    return np.linspace(a, b, n + 1)


def chebyshev_nodes(a, b, n):
    """
    Return n+1 Chebyshev extrema nodes mapped from [-1,1] to [a, b].
    Nodes on [-1,1]: t_k = cos(pi * k / n), k=0,...,n.
    """
    k = np.arange(n + 1)
    t = np.cos(np.pi * k / n)  # Chebyshev extrema on [-1,1]
    x = 0.5 * (a + b) + 0.5 * (b - a) * t
    return x


# -----------------------------
# Simple Lagrange interpolation
# -----------------------------

def lagrange_basis(x_nodes, k, x):
    """
    Compute the k-th Lagrange basis polynomial ℓ_k(x)
    defined by:
        ℓ_k(x_j) = 1 if j == k
        ℓ_k(x_j) = 0 if j != k

    This is the slow but very readable version: double loop.
    """
    xk = x_nodes[k]
    basis = 1.0
    for j, xj in enumerate(x_nodes):
        if j != k:
            basis *= (x - xj) / (xk - xj)
    return basis


def lagrange_interpolant(x_nodes, f_nodes, x):
    """
    Evaluate the Lagrange interpolant at a single point x:
        L(x) = sum_k f_k * ℓ_k(x)
    """
    total = 0.0
    n = len(x_nodes)
    for k in range(n):
        total += f_nodes[k] * lagrange_basis(x_nodes, k, x)
    return total


def evaluate_interpolant(x_nodes, f_nodes, x_eval):
    """
    Evaluate the interpolant at all points in x_eval.
    x_eval can be any 1D array-like.
    """
    x_eval = np.asarray(x_eval)
    y = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        y[i] = lagrange_interpolant(x_nodes, f_nodes, x)
    return y


# -----------------------------
# Helper for error computation
# -----------------------------

def compute_error(f, x_nodes, f_nodes, a, b, num_eval=1000):
    """
    Compute interpolation error on [a,b] for a given function f and nodes.

    Returns:
        x_eval   : evaluation grid
        f_true   : true values f(x_eval)
        f_interp : interpolant values
        err      : |f_true - f_interp|
        max_err  : max_j err[j]
    """
    x_eval = np.linspace(a, b, num_eval)
    f_true = f(x_eval)
    f_interp = evaluate_interpolant(x_nodes, f_nodes, x_eval)
    err = np.abs(f_true - f_interp)
    max_err = np.max(err)
    return x_eval, f_true, f_interp, err, max_err


# -----------------------------
# "Fig. 2" style experiment
# -----------------------------

def run_fig2():
    """
    Conceptual reproduction of Fig. 2:
    - Two rational functions on [-1,1]
    - Interpolated with equally spaced nodes
    - Show how a 'nice' function behaves vs a Runge-like function.
    """
    a, b = -1.0, 1.0

    # Function 1: well-behaved rational
    def f1(x):
        return 1.0 / (1.0 + 0.25 * x**2)

    # Function 2: more severe Runge-type behavior
    def f2(x):
        return 1.0 / (1.0 + 25.0 * x**2)

    ns = [5, 10, 20, 40]  # degrees

    x_plot = np.linspace(a, b, 1000)
    f1_true = f1(x_plot)
    f2_true = f2(x_plot)

    # Figure for f1
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f1_true, 'k-', label="f1(x) true")
    for n in ns:
        x_nodes = equally_spaced_nodes(a, b, n)
        f_nodes = f1(x_nodes)
        f_interp = evaluate_interpolant(x_nodes, f_nodes, x_plot)
        plt.plot(x_plot, f_interp, label=f"n={n}")
    plt.title("Fig. 2 style: Equally spaced interpolation for f1(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig2_f1_interp.png", dpi=200)

    # Figure for f2
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f2_true, 'k-', label="f2(x) true")
    for n in ns:
        x_nodes = equally_spaced_nodes(a, b, n)
        f_nodes = f2(x_nodes)
        f_interp = evaluate_interpolant(x_nodes, f_nodes, x_plot)
        plt.plot(x_plot, f_interp, label=f"n={n}")
    plt.title("Fig. 2 style: Equally spaced interpolation for f2(x) (Runge-like)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1, 2)
    plt.tight_layout()
    plt.savefig("fig2_f2_interp.png", dpi=200)

    # Optional: max error vs n curves for both functions
    max_errs_f1 = []
    max_errs_f2 = []
    for n in ns:
        x_nodes = equally_spaced_nodes(a, b, n)
        f_nodes1 = f1(x_nodes)
        f_nodes2 = f2(x_nodes)
        _, _, _, _, max_e1 = compute_error(f1, x_nodes, f_nodes1, a, b)
        _, _, _, _, max_e2 = compute_error(f2, x_nodes, f_nodes2, a, b)
        max_errs_f1.append(max_e1)
        max_errs_f2.append(max_e2)

    plt.figure(figsize=(8, 5))
    plt.semilogy(ns, max_errs_f1, 'o-', label="f1 max error")
    plt.semilogy(ns, max_errs_f2, 's-', label="f2 max error")
    plt.title("Fig. 2 style: Max error vs degree (equally spaced)")
    plt.xlabel("degree n")
    plt.ylabel("max error (log scale)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_max_error.png", dpi=200)


# -----------------------------
# "Fig. 3" style experiment
# -----------------------------

def run_fig3():
    """
    Conceptual reproduction of Fig. 3:
    - Non-smooth function f(x) = |x| on [-1,1]
    - Compare equally spaced vs Chebyshev nodes at degree n
    """
    a, b = -1.0, 1.0

    def f(x):
        return np.abs(x)

    n = 20  # degree for comparison

    # Nodes
    x_eq = equally_spaced_nodes(a, b, n)
    x_ch = chebyshev_nodes(a, b, n)

    # Sample values
    f_eq_nodes = f(x_eq)
    f_ch_nodes = f(x_ch)

    # Evaluate on fine grid
    x_plot = np.linspace(a, b, 1000)
    f_true = f(x_plot)

    f_eq_interp = evaluate_interpolant(x_eq, f_eq_nodes, x_plot)
    f_ch_interp = evaluate_interpolant(x_ch, f_ch_nodes, x_plot)

    # Plot interpolants vs true function
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f_true, 'k-', label="f(x) = |x| (true)")
    plt.plot(x_plot, f_eq_interp, 'r--', label="Equally spaced nodes")
    plt.plot(x_plot, f_ch_interp, 'b-.', label="Chebyshev nodes")
    plt.title("Fig. 3 style: Interpolation of |x| (n=20)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1,2)
    plt.tight_layout()
    plt.savefig("fig3_interp.png", dpi=200)

    # Plot error curves
    err_eq = np.abs(f_true - f_eq_interp)
    err_ch = np.abs(f_true - f_ch_interp)

    plt.figure(figsize=(8, 5))
    plt.semilogy(x_plot, err_eq, 'r--', label="Equally spaced error")
    plt.semilogy(x_plot, err_ch, 'b-.', label="Chebyshev error")
    plt.title("Fig. 3 style: Error for interpolation of |x| (n=20)")
    plt.xlabel("x")
    plt.ylabel("absolute error (log scale)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("fig3_error.png", dpi=200)


# -----------------------------
# Extra functions experiment
# -----------------------------

def run_extra():
    """
    Apply the same numerical computations to two more functions:

    g1: analytic, with singularities near the real axis (Runge-like)
    g2: low regularity, Hölder but non-differentiable at 0
    """
    a, b = -1.0, 1.0
    ns = [5, 10, 20, 40]

    # Extra function 1: analytic with nearby complex singularities
    # Poles at x = 0.1 ± 0.1i
    def g1(x):
        return 1.0 / (1.0 + 100.0 * (x - 0.1)**2)

    # Extra function 2: non-smooth but continuous (Hölder)
    def g2(x):
        return np.abs(x)**0.3

    # ----- Experiment for g1 -----
    x_plot = np.linspace(a, b, 1000)
    g1_true = g1(x_plot)

    # Compare equally spaced vs Chebyshev (max error vs n)
    max_err_eq_g1 = []
    max_err_ch_g1 = []

    for n in ns:
        # Equally spaced
        x_eq = equally_spaced_nodes(a, b, n)
        g1_eq_nodes = g1(x_eq)
        _, _, _, _, max_e_eq = compute_error(g1, x_eq, g1_eq_nodes, a, b)
        max_err_eq_g1.append(max_e_eq)

        # Chebyshev
        x_ch = chebyshev_nodes(a, b, n)
        g1_ch_nodes = g1(x_ch)
        _, _, _, _, max_e_ch = compute_error(g1, x_ch, g1_ch_nodes, a, b)
        max_err_ch_g1.append(max_e_ch)

    plt.figure(figsize=(8, 5))
    plt.semilogy(ns, max_err_eq_g1, 'ro--', label="Equally spaced")
    plt.semilogy(ns, max_err_ch_g1, 'bs--', label="Chebyshev")
    plt.title("Extra function g1: max error vs degree")
    plt.xlabel("degree n")
    plt.ylabel("max error (log scale)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("extra_g1_max_error.png", dpi=200)

    # ----- Experiment for g2 -----
    x_plot = np.linspace(a, b, 1000)
    g2_true = g2(x_plot)

    max_err_eq_g2 = []
    max_err_ch_g2 = []

    for n in ns:
        # Equally spaced
        x_eq = equally_spaced_nodes(a, b, n)
        g2_eq_nodes = g2(x_eq)
        _, _, _, _, max_e_eq = compute_error(g2, x_eq, g2_eq_nodes, a, b)
        max_err_eq_g2.append(max_e_eq)

        # Chebyshev
        x_ch = chebyshev_nodes(a, b, n)
        g2_ch_nodes = g2(x_ch)
        _, _, _, _, max_e_ch = compute_error(g2, x_ch, g2_ch_nodes, a, b)
        max_err_ch_g2.append(max_e_ch)

    plt.figure(figsize=(8, 5))
    plt.semilogy(ns, max_err_eq_g2, 'ro--', label="Equally spaced")
    plt.semilogy(ns, max_err_ch_g2, 'bs--', label="Chebyshev")
    plt.title("Extra function g2: max error vs degree")
    plt.xlabel("degree n")
    plt.ylabel("max error (log scale)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("extra_g2_max_error.png", dpi=200)


# -----------------------------
# Main dispatch
# -----------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python interpolation_project.py [fig2|fig3|extra]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "fig2":
        run_fig2()
    elif mode == "fig3":
        run_fig3()
    elif mode == "extra":
        run_extra()
    else:
        print(f"Unknown mode '{mode}'. Use one of: fig2, fig3, extra.")
        sys.exit(1)


if __name__ == "__main__":
    main()
