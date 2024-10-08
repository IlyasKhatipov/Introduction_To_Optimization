import numpy as np
from tabulate import tabulate


def simplex_method(C, A, b, optimization="max", precision=1e-5, max_iterations=1000, verbose=True):
    """
    Implements the Simplex method for linear programming problems (maximization or minimization).

    :param C: Coefficients of the objective function.
    :param A: Coefficients matrix of the constraints.
    :param b: Right-hand side vector of the constraints.
    :param optimization: Type of optimization ("max" for maximization, "min" for minimization).
    :param precision: Precision for determining optimality.
    :param max_iterations: Maximum number of iterations to prevent infinite loops.
    :param verbose: If True, displays the tableau at each iteration.
    :return: Tuple containing the optimal variables and the optimal value, or an error message.
    """
    m, n = A.shape

    # Convert minimization to maximization by negating the objective coefficients
    if optimization.lower() == "min":
        C = -C

    # Add slack variables to convert inequalities to equalities
    A = np.hstack([A, np.eye(m)])  # Adding identity matrix for slack variables
    C = np.hstack([C, np.zeros(m)])  # Adding zeros for slack variables in the objective function

    # Initialize basic and non-basic variables
    basic_vars = np.arange(n, n + m)
    non_basic_vars = np.arange(n)

    # Initialize the Simplex tableau
    tableau = np.hstack([A, b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.hstack([-C, np.zeros(1)])])

    iteration = 0
    while iteration < max_iterations:
        if verbose:
            print(f"\nIteration {iteration + 1}:")
            display_tableau(tableau, basic_vars, non_basic_vars, C, m, n)

        # Check for optimality (all coefficients in the objective row >= -precision)
        if np.all(tableau[-1, :-1] >= -precision):
            # Optimal solution found
            x = np.zeros(n + m)
            x[basic_vars] = tableau[:-1, -1]
            # If minimization, convert the optimal value back
            optimal_value = tableau[-1, -1]
            if optimization.lower() == "min":
                optimal_value = -optimal_value
            return x[:n], optimal_value  # Return only the original variables and the objective value

        # Select entering variable (most negative coefficient in the objective row)
        entering_var = np.argmin(tableau[-1, :-1])

        # Check for unboundedness
        if np.all(tableau[:-1, entering_var] <= precision):
            return "The method is not applicable!"  # Unbounded solution

        # Perform the minimum ratio test to select the leaving variable
        ratios = tableau[:-1, -1] / tableau[:-1, entering_var]
        ratios[tableau[:-1, entering_var] <= precision] = np.inf  # Exclude non-positive ratios
        leaving_var = np.argmin(ratios)

        if ratios[leaving_var] == np.inf:
            return "The method is not applicable!"  # Unbounded solution

        # Pivot around the selected element
        pivot = tableau[leaving_var, entering_var]
        tableau[leaving_var, :] /= pivot
        for i in range(m + 1):
            if i != leaving_var:
                tableau[i, :] -= tableau[i, entering_var] * tableau[leaving_var, :]

        # Update basic and non-basic variables
        basic_vars[leaving_var], non_basic_vars[entering_var] = non_basic_vars[entering_var], basic_vars[leaving_var]

        iteration += 1

    return "The method did not converge within the maximum number of iterations."


def display_tableau(tableau, basic_vars, non_basic_vars, C, m, n):
    """
    Displays the current Simplex tableau in a formatted table.

    :param tableau: Current Simplex tableau.
    :param basic_vars: Indices of basic variables.
    :param non_basic_vars: Indices of non-basic variables.
    :param C: Coefficients of the objective function.
    :param m: Number of constraints.
    :param n: Number of original variables.
    """
    headers = ["Basic/Non-Basic"] + [f"x{var + 1}" for var in non_basic_vars] + [f"s{var + 1}" for var in range(m)] + [
        "RHS"]
    table = []
    for i in range(m):
        row = [f"x{basic_vars[i] + 1}"] + [f"{tableau[i, j]:.2f}" for j in range(n + m)] + [f"{tableau[i, -1]:.2f}"]
        table.append(row)
    # Objective function row
    obj_row = ["Z"] + [f"{tableau[-1, j]:.2f}" for j in range(n + m)] + [f"{tableau[-1, -1]:.2f}"]
    table.append(obj_row)
    print(tabulate(table, headers=headers, tablefmt="grid"))


def get_user_input():
    """
    Collects input data from the user with prompts.

    :return: Tuple containing C, A, b, precision, and optimization type.
    """
    print("Welcome to the Simplex Method Linear Programming Solver!")
    print("The problem should be in the form:")
    print("Maximize or Minimize Z = C1*x1 + C2*x2 + ... + Cn*xn")
    print("Subject to:")
    print("a11*x1 + a12*x2 + ... + a1n*xn <= b1")
    print("a21*x1 + a22*x2 + ... + a2n*xn <= b2")
    print("...")
    print("am1*x1 + am2*x2 + ... + amn*xn <= bm")
    print()

    # Choose optimization type
    while True:
        optimization = input(
            "Choose optimization type ('max' for maximization, 'min' for minimization): ").strip().lower()
        if optimization in ["max", "min"]:
            break
        else:
            print("Invalid input. Please enter 'max' or 'min'.")

    # Input objective function coefficients
    while True:
        try:
            C = list(map(float, input(
                "Enter the objective function coefficients C separated by spaces (e.g., 3 2): ").strip().split()))
            if len(C) == 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")

    n = len(C)

    # Input number of constraints
    while True:
        try:
            m = int(input("Enter the number of constraints (m): "))
            if m <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

    A = []
    b = []
    for i in range(m):
        while True:
            try:
                constraint = list(map(float, input(
                    f"Enter the coefficients for constraint {i + 1} separated by spaces (must be {n} numbers): ").strip().split()))
                if len(constraint) != n:
                    raise ValueError
                break
            except ValueError:
                print(f"Invalid input. Please enter exactly {n} numbers separated by spaces.")
        A.append(constraint)
        while True:
            try:
                bi = float(input(f"Enter the right-hand side for constraint {i + 1} (b{i + 1}): "))
                b.append(bi)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Input precision
    while True:
        try:
            precision = float(input("Enter the precision (e.g., 1e-5): "))
            if precision <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive number.")

    return np.array(C), np.array(A), np.array(b), precision, optimization


def main():
    # Get user input
    C, A, b, precision, optimization = get_user_input()

    # Call the Simplex method
    result = simplex_method(C, A, b, optimization, precision)

    # Display the results
    print("\nResult:")
    if isinstance(result, tuple):
        x_optimal, optimal_value = result
        for idx, x in enumerate(x_optimal, start=1):
            print(f"x{idx} = {x:.4f}")
        if optimization.lower() == "max":
            print(f"Maximum value of the objective function: {optimal_value:.4f}")
        else:
            print(f"Minimum value of the objective function: {optimal_value:.4f}")
    else:
        print(result)


if __name__ == "__main__":
    main()
