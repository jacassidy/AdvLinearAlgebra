import numpy as np
import sympy as sp
from sympy import Matrix
import matplotlib.pyplot as plt

def powX(n):
    """Returns a function that computes x^n."""
    return lambda x: x**n

#***Parameters***
#****************
print_solution_vector = True
print_area_gap = True
display_graph = True

POINTS_OF_INTEGRATION = 1000
BASIS_VECTOR_SIZE = 6  # polynomial degree will be BASIS_VECTOR_SIZE - 1
bounds = [-np.pi, np.pi]

inf_dim_vector_to_project = np.sin

# Taylor settings
TAYLOR_CENTER = 0.0  # Maclaurin when 0.0; change to e.g. (bounds[0]+bounds[1])/2 for midpoint
#********************

# Generate basis vectors based on the specified size
basis_vectors = [powX(i) for i in range(BASIS_VECTOR_SIZE)]

def RealL2InnerProduct(a, b):
    x = np.linspace(bounds[0], bounds[1], POINTS_OF_INTEGRATION)
    a_values = a(x)
    b_values = b(x)
    return np.trapezoid(a_values * b_values, x)

def inner_product(a, b):
    return RealL2InnerProduct(a, b)

def numpy_func_to_sympy_expr(np_func, x):
    """
    Map the numpy function you’re projecting to a SymPy expression for Taylor.
    Extend this mapping if you swap inf_dim_vector_to_project to something else.
    """
    if np_func is np.sin:
        return sp.sin(x)
    if np_func is np.cos:
        return sp.cos(x)
    if np_func is np.exp:
        return sp.exp(x)
    if np_func is np.log:
        return sp.log(x)
    raise ValueError("No SymPy mapping for inf_dim_vector_to_project. Add one in numpy_func_to_sympy_expr().")

def make_taylor_callable(np_func, degree, center):
    x = sp.Symbol('x')
    expr = numpy_func_to_sympy_expr(np_func, x)
    poly = sp.series(expr, x, center, degree + 1).removeO()
    return sp.lambdify(x, sp.expand(poly), "numpy")

def main():
    basis_size = len(basis_vectors)

    matrix = np.zeros((basis_size, basis_size), dtype=float)
    vector = np.zeros(basis_size, dtype=float)

    for i in range(basis_size):
        for j in range(basis_size):
            matrix[i][j] = inner_product(basis_vectors[j], basis_vectors[i])
        vector[i] = inner_product(inf_dim_vector_to_project, basis_vectors[i])

    augmented_matrix = Matrix(np.column_stack((matrix, vector)))
    rref_matrix, pivot_columns = augmented_matrix.rref()

    solution_vector = rref_matrix[:, -1]

    if print_solution_vector:
        print("The solution vector c is:\nA =", solution_vector)

    return solution_vector

def f(coeffs, x):
    # coeffs might be a SymPy Matrix; convert to numeric 1D array
    a = np.asarray([float(v) for v in coeffs], dtype=float)
    # Horner would be faster, but this is clear and fine for small degrees
    return sum(a[i] * x**i for i in range(len(a)))

if __name__ == "__main__":
    coefficients = main()

    degree = BASIS_VECTOR_SIZE - 1
    x_values = np.linspace(bounds[0], bounds[1], POINTS_OF_INTEGRATION)

    projected_values = f(coefficients, x_values)
    original_values = inf_dim_vector_to_project(x_values)

    # Taylor polynomial values (same degree)
    taylor_func = make_taylor_callable(inf_dim_vector_to_project, degree, TAYLOR_CENTER)
    taylor_values = taylor_func(x_values)

    if display_graph:
        plt.plot(x_values, original_values, label='Original')
        plt.plot(x_values, projected_values, label='Projected', linestyle='--')
        plt.plot(x_values, taylor_values, label='Taylor', linestyle=':')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Original vs Projected vs Taylor (degree {degree})')
        plt.show()

    if print_area_gap:
        area_gap = np.trapezoid(np.abs(original_values - projected_values), x_values)
        print(f"The area of the gap between the projected and original function is: {area_gap}")
