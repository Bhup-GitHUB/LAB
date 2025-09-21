import copy

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

def is_consistent(assignment, var, value):
    for constraint in assignment:
        if constraint != var:
            if not check_constraint(constraint, assignment[constraint], var, value):
                return False
    return True

def check_constraint(var1, val1, var2, val2):
    row1, col1 = var1
    row2, col2 = var2
    
    if row1 == row2 or col1 == col2:
        return False
    
    if abs(row1 - row2) == abs(col1 - col2):
        return False
    
    return True

def backtrack_search(csp, assignment={}):
    if len(assignment) == len(csp.variables):
        return assignment
    
    var = select_unassigned_variable(csp, assignment)
    
    for value in order_domain_values(csp, var, assignment):
        if is_consistent(assignment, var, value):
            assignment[var] = value
            result = backtrack_search(csp, assignment)
            if result is not None:
                return result
            del assignment[var]
    
    return None

def select_unassigned_variable(csp, assignment):
    for var in csp.variables:
        if var not in assignment:
            return var
    return None

def order_domain_values(csp, var, assignment):
    return csp.domains[var]

def solve_8_queens():
    variables = [(i, j) for i in range(8) for j in range(8)]
    domains = {}
    
    for var in variables:
        domains[var] = [True, False]
    
    constraints = []
    
    csp = CSP(variables, domains, constraints)
    
    assignment = {}
    
    queens_placed = 0
    for row in range(8):
        for col in range(8):
            if queens_placed >= 8:
                break
            if is_consistent(assignment, (row, col), True):
                assignment[(row, col)] = True
                queens_placed += 1
                break
    
    if len(assignment) == 8:
        return assignment
    else:
        return None

def solve_8_queens_recursive():
    board = [-1] * 8
    solutions = []
    
    def is_safe(row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def place_queen(row):
        if row == 8:
            solutions.append(board[:])
            return
        
        for col in range(8):
            if is_safe(row, col):
                board[row] = col
                place_queen(row + 1)
                board[row] = -1
    
    place_queen(0)
    return solutions

def print_board(solution):
    if not solution:
        print("No solution found")
        return
    
    print("8-Queens Solution:")
    print("  a b c d e f g h")
    
    for row in range(8):
        print(f"{8-row} ", end="")
        for col in range(8):
            if solution[row] == col:
                print("Q ", end="")
            else:
                print(". ", end="")
        print(f"{8-row}")
    
    print("  a b c d e f g h")

def print_coordinates(solution):
    if not solution:
        print("No solution found")
        return
    
    print("\nQueen positions:")
    for row in range(8):
        col = solution[row]
        col_letter = chr(ord('a') + col)
        row_number = 8 - row
        print(f"Row {row_number}, Column {col_letter}")

def main():
    print("Eight-Queens Problem - Constraint Satisfaction Problem")
    print("=" * 60)
    
    print("Problem Definition:")
    print("- Variables: 8 queens (one per row)")
    print("- Domain: 8 columns (a-h)")
    print("- Constraints: No two queens can attack each other")
    print("  - No two queens in same row")
    print("  - No two queens in same column") 
    print("  - No two queens in same diagonal")
    print()
    
    solutions = solve_8_queens_recursive()
    
    if solutions:
        print(f"Found {len(solutions)} solutions")
        print("\nFirst solution:")
        print_board(solutions[0])
        print_coordinates(solutions[0])
        
        print("\nSecond solution:")
        if len(solutions) > 1:
            print_board(solutions[1])
            print_coordinates(solutions[1])
        
        print("\nThird solution:")
        if len(solutions) > 2:
            print_board(solutions[2])
            print_coordinates(solutions[2])
        
        print(f"\nTotal solutions found: {len(solutions)}")
        
        print("\nCSP Implementation Details:")
        print("- Variables: Each row represents a queen")
        print("- Domain: Column positions (0-7)")
        print("- Constraints: No attacking positions")
        print("- Algorithm: Backtracking with constraint checking")
        
    else:
        print("No solutions found")

if __name__ == "__main__":
    main()
