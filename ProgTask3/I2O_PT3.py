import numpy as np

def check_balance(supply, demand):
    if sum(supply) != sum(demand):
        return False, "The problem is not balanced!"
    return True, "Balanced problem."

def print_table(cost_matrix, supply, demand):
    print("Parameter Table of Transportation Problem:")
    print("Cost Matrix:")
    print(cost_matrix)
    print("Supply:", supply)
    print("Demand:", demand)

def north_west_corner(supply, demand):
    supply = supply.copy()
    demand = demand.copy()
    rows, cols = len(supply), len(demand)
    allocation = np.zeros((rows, cols))
    
    i, j = 0, 0
    while i < rows and j < cols:
        min_value = min(supply[i], demand[j])
        allocation[i][j] = min_value
        supply[i] -= min_value
        demand[j] -= min_value
        if supply[i] == 0:
            i += 1
        else:
            j += 1
    return allocation

def vogel_approximation(supply, demand, cost_matrix):
    supply = supply.copy()
    demand = demand.copy()
    allocation = np.zeros_like(cost_matrix, dtype=float)
    cost_matrix = cost_matrix.astype(float)  # Convert to float to handle infinity
    
    while sum(supply) > 0 and sum(demand) > 0:
        # Calculate penalties
        row_penalty = []
        for row in cost_matrix:
            non_inf_values = row[row < float('inf')]
            if len(non_inf_values) > 1:
                sorted_vals = np.sort(non_inf_values)
                row_penalty.append(sorted_vals[1] - sorted_vals[0])
            else:
                row_penalty.append(float('inf'))
        
        col_penalty = []
        for col in cost_matrix.T:
            non_inf_values = col[col < float('inf')]
            if len(non_inf_values) > 1:
                sorted_vals = np.sort(non_inf_values)
                col_penalty.append(sorted_vals[1] - sorted_vals[0])
            else:
                col_penalty.append(float('inf'))
        
        print("Row Penalty:", row_penalty)  # Debug
        print("Column Penalty:", col_penalty)  # Debug
        
        # Determine the maximum penalty and corresponding row or column
        max_row_penalty = min(row_penalty)
        max_col_penalty = min(col_penalty)
                # Find the cell with the highest penalty, choosing row or column accordingly
        if min(row_penalty) <= min(col_penalty):
            row = np.argmin(row_penalty)
            col = np.argmin(cost_matrix[row])
            print(f"Chosen cell from row {row} with column {col}")  # Debug
        else:
            col = np.argmin(col_penalty)
            row = np.argmin(cost_matrix[:, col])
            print(f"Chosen cell from column {col} with row {row}")  # Debug
        
        # Allocate the minimum of supply or demand to the chosen cell
        min_value = min(supply[row], demand[col])
        allocation[row][col] = min_value
        supply[row] -= min_value
        demand[col] -= min_value
        print(f"Allocating {min_value} units to cell ({row}, {col})")  # Debug
        
        # Mark the exhausted row or column by setting their costs to infinity
        if supply[row] == 0:
            cost_matrix[row, :] = float('inf')
            print(f"Row {row} exhausted")  # Debug
        if demand[col] == 0:
            cost_matrix[:, col] = float('inf')
            print(f"Column {col} exhausted")  # Debug
        
        print("Current Allocation Matrix:\n", allocation)  # Debug
    
    return allocation

# All previously defined functions are the same, with vogel_approximation replaced by the new code

def transportation_problem(supply, cost_matrix, demand):
    balance, message = check_balance(supply, demand)
    if not balance:
        print(message)
        return
    else:
        print_table(cost_matrix, supply, demand)
        
        print("\nNorth-West Corner Solution:")
        nw_solution = north_west_corner(supply, demand)
        print(nw_solution)
        
        print("\nVogel’s Approximation Solution:")
        vogel_solution = vogel_approximation(supply, demand, cost_matrix)
        print(vogel_solution)
        
        print("\nRussell’s Approximation Solution:")
        russell_solution = russell_approximation(supply, demand, cost_matrix)
        print(russell_solution)

# Input example
S = [20, 30, 25]   # Supply vector
D = [10, 25, 15, 25]  # Demand vector
C = np.array([
    [8, 6, 10, 9],
    [9, 12, 13, 7],
    [14, 9, 16, 5]
])  # Cost matrix

transportation_problem(S, C, D)
