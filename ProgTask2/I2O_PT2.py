import numpy as np
from numpy.linalg import inv, norm

def interior_point_method(C, A, x_init, epsilon=1e-6, alpha=0.5):
    x = np.array(x_init, float)
    i = 1
    
    while True:
        v = x.copy()
        D = np.diag(x)  # Diagonal matrix of x
        AA = np.dot(A, D)  # A * D
        cc = np.dot(D, C)  # D * C
        I = np.eye(len(C))  # Identity matrix

        # Compute F and its inverse
        F = np.dot(AA, np.transpose(AA))  # A * D * (A * D)^T
        try:
            FI = inv(F)  # F^(-1)
        except np.linalg.LinAlgError:
            return "The method is not applicable!"  # F is singular

        # Compute projection matrix P
        H = np.dot(np.transpose(AA), FI)  # (A*D)^T * F^(-1)
        P = np.subtract(I, np.dot(H, AA))  # I - H * (A * D)

        # Compute cp and find the step size nu
        cp = np.dot(P, cc)  # P * (D * C)
        nu = np.absolute(np.min(cp))  # Find minimum absolute value in cp

        if nu == 0:
            return "The problem does not have a solution!"  # Problem is infeasible

        # Update x using alpha and nu
        y = np.add(np.ones(len(C), float), (alpha / nu) * cp)  # y = 1 + (alpha / nu) * cp
        yy = np.dot(D, y)  # Update x

        x = yy  # Set x for the next iteration

        # Print intermediate results for first 4 iterations
        if i <= 4:
            print(f"In iteration {i}, we have x = {x}\n")

        # Stop if the change is smaller than epsilon
        if norm(np.subtract(yy, v), ord=2) < epsilon:
            break

        i += 1

    return x, np.dot(C, x)

def run_test_case(test_number, C, A, x_init, epsilon, expected_alpha_05, expected_alpha_09):
    print(f"=== Test Case {test_number} ===")
    print("Parameters:")
    print(f"C = {C}")
    print(f"A =\n{A}")
    print(f"x_init = {x_init}")
    print(f"epsilon = {epsilon}\n")

    # Run for alpha = 0.5
    print("Results for alpha = 0.5:")
    result_alpha_05 = interior_point_method(C, A, x_init, epsilon, alpha=0.5)
    print(f"Final x = {result_alpha_05[0]}, Optimal value = {result_alpha_05[1]}")
    print(f"Expected x ≈ {expected_alpha_05[0]}, Expected Optimal value ≈ {expected_alpha_05[1]}\n")

    # Run for alpha = 0.9
    print("Results for alpha = 0.9:")
    result_alpha_09 = interior_point_method(C, A, x_init, epsilon, alpha=0.9)
    print(f"Final x = {result_alpha_09[0]}, Optimal value = {result_alpha_09[1]}")
    print(f"Expected x ≈ {expected_alpha_09[0]}, Expected Optimal value ≈ {expected_alpha_09[1]}\n")
    print("========================================\n")


def main():
    # Define all test cases as a list of dictionaries
    test_cases = [
        {
            "number": 1,
            "description": "Standard Feasible LP with Unique Solution",
            "C": np.array([-3, -2, 0, 0], float),
            "A": np.array([[1, 2, 1, 0], [3, 2, 0, 1]], float),
            "x_init": [1, 1, 1, 1],
            "epsilon": 1e-6,
            "expected_alpha_05": ([2, 1, 0, 3], 8),
            "expected_alpha_09": ([2, 1, 0, 3], 8)
        },
        {
            "number": 2,
            "description": "Infeasible LP",
            "C": np.array([-1, -1, 0, 0], float),
            "A": np.array([[1, 1, 1, 0], [-1, -1, 0, 1]], float),
            "x_init": [1, 1, 1, 1],
            "epsilon": 1e-6,
            "expected_alpha_05": ("The problem does not have a solution!", None),
            "expected_alpha_09": ("The problem does not have a solution!", None)
        },
        {
            "number": 3,
            "description": "LP with Multiple Optimal Solutions",
            "C": np.array([-2, -3, 0, 0], float),
            "A": np.array([[1, 1, 1, 0], [2, 3, 0, 1]], float),
            "x_init": [1, 1, 1, 1],
            "epsilon": 1e-6,
            "expected_alpha_05": ("Multiple optimal solutions possible", "Multiple optimal solutions possible"),
            "expected_alpha_09": ("Multiple optimal solutions possible", "Multiple optimal solutions possible")
        },
        {
            "number": 4,
            "description": "LP Causing Singular F Matrix",
            "C": np.array([-1, -1, 0, 0], float),
            "A": np.array([[1, 1, 1, 0], [2, 2, 0, 1]], float),
            "x_init": [1, 1, 1, 1],
            "epsilon": 1e-6,
            "expected_alpha_05": ("The method is not applicable!", None),
            "expected_alpha_09": ("The method is not applicable!", None)
        },
        {
            "number": 5,
            "description": "Higher-Dimensional LP",
            "C": np.array([-5, -4, -3, 0, 0, 0], float),
            "A": np.array([
                [1, 1, 1, 1, 0, 0],
                [2, 2, 1, 0, 1, 0],
                [2, 3, 2, 0, 0, 1]
            ], float),
            "x_init": [1, 1, 1, 1, 1, 1],
            "epsilon": 1e-6,
            "expected_alpha_05": ([4, 2, 4, 2, 0, 0], 40),
            "expected_alpha_09": ([4, 2, 4, 2, 0, 0], 40)
        },
        {
            "number": 6,
            "description": "LP with Alpha Affecting Convergence",
            "C": np.array([-1, -2, 0, 0], float),
            "A": np.array([[1, 1, 1, 0], [1, 3, 0, 1]], float),
            "x_init": [1, 1, 1, 1],
            "epsilon": 1e-6,
            "expected_alpha_05": ([3, 2, 0, 0], 7),
            "expected_alpha_09": ([3, 2, 0, 0], 7)
        }
    ]

    # Iterate over each test case and run them
    for test in test_cases:
        run_test_case(
            test_number=test["number"],
            C=test["C"],
            A=test["A"],
            x_init=test["x_init"],
            epsilon=test["epsilon"],
            expected_alpha_05=test["expected_alpha_05"],
            expected_alpha_09=test["expected_alpha_09"]
        )

if __name__ == "__main__":
    main()
