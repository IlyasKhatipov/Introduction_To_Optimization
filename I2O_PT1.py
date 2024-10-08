import numpy as np

# Simplex method implementation for maximization
def simplex_method(C, A, b, precision=1e-5):
    m, n = A.shape

    # Add slack variables to transform inequalities into equalities
    A = np.hstack([A, np.eye(m)])  # Add identity matrix to represent slack variables
    C = np.hstack([C, np.zeros(m)])  # Add zeros to the cost function for slack variables

    # Initialize basic and non-basic variables
    basic_vars = np.arange(n, n + m)
    non_basic_vars = np.arange(n)

    # Initialize the tableau
    tableau = np.hstack([A, b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.hstack([-C, np.zeros(1)])])

    # Start the iteration process
    while True:
        # Check if the current solution is optimal (all cost row coefficients are non-negative)
        if np.all(tableau[-1, :-1] >= -precision):
            # Optimal solution found
            x = np.zeros(n + m)
            x[basic_vars] = tableau[:-1, -1]
            return x[:n], tableau[-1, -1]  # Return decision variables and objective function value

        # Choose entering variable (most negative coefficient in the cost row)
        entering_var = np.argmin(tableau[-1, :-1])

        # Check for unboundedness
        if np.all(tableau[:-1, entering_var] <= precision):
            return "The method is not applicable!"  # Unbounded solution

        # Choose leaving variable using the minimum ratio test
        ratios = tableau[:-1, -1] / tableau[:-1, entering_var]
        leaving_var = np.argmin(np.where(ratios > precision, ratios, np.inf))

        # Pivot around the chosen variables
        tableau[leaving_var, :] /= tableau[leaving_var, entering_var]
        for i in range(m + 1):
            if i != leaving_var:
                tableau[i, :] -= tableau[i, entering_var] * tableau[leaving_var, :]

        # Update basic and non-basic variables
        basic_vars[leaving_var], non_basic_vars[entering_var] = non_basic_vars[entering_var], basic_vars[leaving_var]


# Example usage
C = np.array([3, 2])  # Coefficients of the objective function
A = np.array([[1, 2], [3, 2]])  # Coefficients of constraints
b = np.array([6, 12])  # Right-hand side values

x_optimal, max_value = simplex_method(C, A, b)
print("Optimal decision variables:", x_optimal)
print("Maximum value of the objective function:", max_value)
