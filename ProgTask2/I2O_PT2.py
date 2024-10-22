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

# Define the linear programming problem
C = np.array([-3, -2, 0, 0], float)  # Coefficients of the objective function (-3x1 - 2x2 for maximization)
A = np.array([[1, 2, 1, 0], [3, 2, 0, 1]], float)  # Constraint matrix with slack variables
x_init = [1, 1, 1, 1]  # Initial guess for x, including slack variables

# Set epsilon for stopping criteria
epsilon = 1e-6

# Run the algorithm for alpha = 0.5
print("Results for alpha = 0.5:")
result_alpha_05 = interior_point_method(C, A, x_init, epsilon, alpha=0.5)
print(f"Final x = {result_alpha_05[0]}, Optimal value = {result_alpha_05[1]}\n")

# Run the algorithm for alpha = 0.9
print("Results for alpha = 0.9:")
result_alpha_09 = interior_point_method(C, A, x_init, epsilon, alpha=0.9)
print(f"Final x = {result_alpha_09[0]}, Optimal value = {result_alpha_09[1]}")
