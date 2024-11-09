import numpy as np


def get_test_case(case_number):
    if case_number == 1:
        supply = [20, 30, 25]
        demand = [10, 10, 20, 35]
        costs = [
            [8, 6, 10, 9],
            [9, 12, 13, 7],
            [14, 9, 16, 5]
        ]
    elif case_number == 2:
        supply = [15, 25, 20]
        demand = [10, 20, 15, 15]
        costs = [
            [4, 8, 8, 6],
            [6, 5, 7, 4],
            [8, 7, 6, 5]
        ]
    elif case_number == 3:
        supply = [10, 35, 25]
        demand = [20, 15, 10, 25]
        costs = [
            [3, 6, 9, 12],
            [4, 8, 12, 16],
            [5, 10, 15, 20]
        ]
    elif case_number == 4:
        supply = [40, 20, 30]
        demand = [25, 35, 15, 15]
        costs = [
            [2, 3, 1, 4],
            [5, 7, 6, 2],
            [8, 5, 9, 3]
        ]
    else:
        supply, demand, costs = None, None, None
    return supply, demand, costs


def get_input_data():
    """Get input data from user for supply, demand, and cost matrix."""
    print("Welcome to the Transportation Problem Solver!")
    print("Please enter the required data for the problem.\n")

    # Get supply vector
    supply = []
    num_sources = int(input("Enter the number of sources: "))
    for i in range(num_sources):
        s = float(input(f"Enter supply for source S{i}: "))
        supply.append(s)

    # Get demand vector
    demand = []
    num_destinations = int(input("Enter the number of destinations: "))
    for j in range(num_destinations):
        d = float(input(f"Enter demand for destination D{j}: "))
        demand.append(d)

    # Ensure balance
    if sum(supply) != sum(demand):
        print("\nError: Total supply and demand must be equal for a balanced problem.")
        print(f"Total supply: {sum(supply)}, Total demand: {sum(demand)}")
        return None, None, None

    # Get cost matrix
    print("\nEnter the cost matrix (each entry is the cost from a source to a destination):")
    costs = []
    for i in range(num_sources):
        row = []
        for j in range(num_destinations):
            cost = float(input(f"Enter cost from S{i} to D{j}: "))
            row.append(cost)
        costs.append(row)

    print("\nThank you! Here is the data you entered:")
    print_problem_table(supply, demand, costs)
    return supply, demand, costs


def check_balance(supply, demand):
    """Check if the supply and demand vectors are balanced."""
    return sum(supply) == sum(demand)


def format_matrix(matrix, row_labels=None, col_labels=None):
    """Pretty-print a matrix with optional row and column labels."""
    output = ""
    if col_labels:
        output += "      " + " ".join(f"{label:^8}" for label in col_labels) + "\n"
    for i, row in enumerate(matrix):
        row_label = f"{row_labels[i]:<4}" if row_labels else ""
        output += row_label + " " + " ".join(f"{val:^8.2f}" for val in row) + "\n"
    return output


def print_problem_table(supply, demand, costs):
    """Print the transportation problem input table."""
    print("\nTransportation Problem Input Table:")
    print("Cost Matrix (C):")
    print(format_matrix(costs, row_labels=[f"S{i}" for i in range(len(supply))],
                        col_labels=[f"D{j}" for j in range(len(demand))]))
    print("Supply (S):", supply)
    print("Demand (D):", demand)
    print("\n" + "=" * 50)


def north_west_corner(supply, demand):
    """North-West Corner method for initial feasible solution with semi-steps."""
    supply = supply.copy()
    demand = demand.copy()
    allocation = np.zeros((len(supply), len(demand)))

    print("\nStarting North-West Corner Method...")
    i, j = 0, 0
    while i < len(supply) and j < len(demand):
        allocation_amount = min(supply[i], demand[j])
        allocation[i][j] = allocation_amount
        supply[i] -= allocation_amount
        demand[j] -= allocation_amount
        print(f"Allocated {allocation_amount} units to cell (S{i}, D{j}).")
        print(f"Remaining supply: {supply}, Remaining demand: {demand}")
        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1
    print("\nFinal Allocation using North-West Corner Method:")
    print(format_matrix(allocation, row_labels=[f"S{i}" for i in range(len(supply))],
                        col_labels=[f"D{j}" for j in range(len(demand))]))
    return allocation


def vogel_approximation(supply, demand, costs):
    """Vogel's Approximation Method with semi-steps."""
    supply = supply.copy()
    demand = demand.copy()
    allocation = np.zeros((len(supply), len(demand)))
    costs = np.array(costs)

    print("\nStarting Vogel's Approximation Method...")
    while sum(supply) > 0 and sum(demand) > 0:
        row_penalties = []
        col_penalties = []

        for i, s in enumerate(supply):
            if s > 0:
                row = sorted([costs[i][j] for j in range(len(demand)) if demand[j] > 0])
                row_penalties.append(row[1] - row[0] if len(row) > 1 else float('inf'))
            else:
                row_penalties.append(-1)

        for j, d in enumerate(demand):
            if d > 0:
                col = sorted([costs[i][j] for i in range(len(supply)) if supply[i] > 0])
                col_penalties.append(col[1] - col[0] if len(col) > 1 else float('inf'))
            else:
                col_penalties.append(-1)

        row_max = max(row_penalties)
        col_max = max(col_penalties)

        if row_max > col_max:
            i = row_penalties.index(row_max)
            j = min((costs[i][j], j) for j in range(len(demand)) if demand[j] > 0)[1]
            penalty_type = "row"
        else:
            j = col_penalties.index(col_max)
            i = min((costs[i][j], i) for i in range(len(supply)) if supply[i] > 0)[1]
            penalty_type = "column"

        allocation_amount = min(supply[i], demand[j])
        allocation[i][j] = allocation_amount
        supply[i] -= allocation_amount
        demand[j] -= allocation_amount
        print(f"Allocated {allocation_amount} units to cell (S{i}, D{j}) by {penalty_type} penalty.")
        print(f"Remaining supply: {supply}, Remaining demand: {demand}")
        print()

    print("\nFinal Allocation using Vogel's Approximation Method:")
    print(format_matrix(allocation, row_labels=[f"S{i}" for i in range(len(supply))],
                        col_labels=[f"D{j}" for j in range(len(demand))]))
    return allocation


def russell_approximation(supply, demand, costs):
    """Russell's Approximation Method with semi-steps."""
    supply = supply.copy()
    demand = demand.copy()
    allocation = np.zeros((len(supply), len(demand)))
    costs = np.array(costs, dtype=float)

    print("\nStarting Russell's Approximation Method...")

    row_potentials = np.min(costs, axis=1)
    col_potentials = np.min(costs - row_potentials[:, None], axis=0)

    differences = costs - (row_potentials[:, None] + col_potentials)

    while sum(supply) > 0 and sum(demand) > 0:

        min_diff = float('inf')
        min_cell = (-1, -1)

        for i in range(len(supply)):
            for j in range(len(demand)):
                if supply[i] > 0 and demand[j] > 0 and differences[i, j] < min_diff:
                    min_diff = differences[i, j]
                    min_cell = (i, j)

        i, j = min_cell
        allocation_amount = min(supply[i], demand[j])
        allocation[i][j] = allocation_amount
        supply[i] -= allocation_amount
        demand[j] -= allocation_amount
        print(
            f"Allocated {allocation_amount} units to cell (S{i}, D{j}) with cost {costs[i][j]} and difference {min_diff}.")
        print(f"Remaining supply: {supply}, Remaining demand: {demand}")
        print()

        large_value = 1e9
        if supply[i] == 0:
            differences[i, :] = large_value
        if demand[j] == 0:
            differences[:, j] = large_value

    print("\nFinal Allocation using Russell's Approximation Method:")
    print(format_matrix(allocation, row_labels=[f"S{i}" for i in range(len(supply))],
                        col_labels=[f"D{j}" for j in range(len(demand))]))
    return allocation


def solve_transportation_problem(supply, demand, costs):
    if not check_balance(supply, demand):
        print("The problem is not balanced!")
        return

    print_problem_table(supply, demand, costs)

    north_west_corner(supply, demand)
    print("=" * 50)

    vogel_approximation(supply, demand, costs)
    print("=" * 50)

    russell_approximation(supply, demand, costs)
    print("\n" + "=" * 50)


def main():
    print("Transportation Problem Solver")
    print("Select a test case to run:")
    print("1. Test Case 1")
    print("2. Test Case 2")
    print("3. Test Case 3")
    print("4. Test Case 4")
    print("5. Enter your own data")

    choice = int(input("Enter your choice (1-5): "))

    if choice >= 1 and choice <= 4:
        supply, demand, costs = get_test_case(choice)
        solve_transportation_problem(supply, demand, costs)
    elif choice == 5:
        supply, demand, costs = get_input_data()
        if supply and demand and costs:
            solve_transportation_problem(supply, demand, costs)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
