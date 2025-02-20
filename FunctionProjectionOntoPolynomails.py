import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt

def powX(n):
    """
    Returns a function that computes x^n.
    """
    return lambda x: x**n

#***Parameters***
#****************

print_solution_vector=True
print_area_gap=True
display_graph=True

POINTS_OF_INTEGRATION = 1000
BASIS_VECTOR_SIZE = 70  # Specify the size of the basis vector array

bounds = [-np.pi, np.pi]

inf_dim_vector_to_project = np.cos

#***End Parameters***
#********************

# Generate basis vectors based on the specified size
basis_vectors = [powX(i) for i in range(BASIS_VECTOR_SIZE)]

def inner_product(a, b):
    return RealL2InnerProduct(a, b)  # L2 inner product

def RealL2InnerProduct(a, b):
    x = np.linspace(bounds[0], bounds[1], POINTS_OF_INTEGRATION)  # Generate 1000 points between the bounds

    # Evaluate the functions a and b at the points x
    a_values = a(x)
    b_values = b(x)

    """
    Computes the L2 inner product of two functions a and b over the domain.
    """
    return np.trapezoid(a_values * b_values, x)  # Using trapezoidal rule for numerical integration

def main():
    basis_size = len(basis_vectors)

    matrix = np.zeros((basis_size, basis_size))
    vector = np.zeros(basis_size)

    for i in range(basis_size):
        for j in range(basis_size):
            matrix[i][j] = inner_product(basis_vectors[j], basis_vectors[i])

        vector[i] = inner_product(inf_dim_vector_to_project, basis_vectors[i])

    # Convert to sympy Matrix for RREF computation
    augmented_matrix = Matrix(np.column_stack((matrix, vector)))
    rref_matrix, pivot_columns = augmented_matrix.rref()

    # Extract the solution vector c from the RREF matrix
    solution_vector = rref_matrix[:, -1]

    # Truncate the solution vector to 7 decimal places
    truncated_solution_vector = [round(float(val), 7) for val in solution_vector]

    if print_solution_vector:
        print("The solution vector c is: \nA =", truncated_solution_vector)

    return solution_vector

def f(a, x):
    return sum(a[i] * x**i for i in range(len(a)))

if __name__ == "__main__":
    coefficients = main()
    x_values = np.linspace(bounds[0], bounds[1], POINTS_OF_INTEGRATION)
    projected_values = f(coefficients, x_values)
    original_values = inf_dim_vector_to_project(x_values)

    if display_graph:
        plt.plot(x_values, original_values, label='Original Function (sin(x))')
        plt.plot(x_values, projected_values, label='Projected Function', linestyle='--')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Comparison of Original and Projected Functions')
        plt.show()

    if print_area_gap:
        area_gap = np.trapezoid(np.abs(original_values - projected_values), x_values)
        print(f"The area of the gap between the projected and original function is: {area_gap}")

