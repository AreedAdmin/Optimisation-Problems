"""
Solves the 8x8 knight's dominating set problem using Integer Linear Programming.

This script finds the minimum number of knights required to "control"
every square on an 8x8 chessboard.

Requires the 'pulp' library:
pip install pulp
"""

import pulp

def solve_knight_problem():
    """
    Formulates and solves the knight's dominating set problem.
    
    Prints the solution (status, minimum knights, and one possible placement).
    Returns the minimum number of knights.
    """
    prob = pulp.LpProblem("Knight_Dominating_Set", pulp.LpMinimize)
    
    board_size = 8
    
    squares = [(i, j) for i in range(board_size) for j in range(board_size)]
    
    x = pulp.LpVariable.dicts("x", squares, 0, 1, pulp.LpBinary)
    
    prob += pulp.lpSum([x[s] for s in squares]), "Total_Knights"
    
    knight_moves = [
        (1, 2), (1, -2), (-1, 2), (-1, -2),
        (2, 1), (2, -1), (-2, 1), (-2, -1)
    ]
    
    for r in range(board_size):
        for c in range(board_size):
            controlling_squares = []
            
            controlling_squares.append(x[(r, c)])
            
            for dr, dc in knight_moves:
                nr, nc = r + dr, c + dc  
                
                if (nr, nc) in squares:
                    controlling_squares.append(x[(nr, nc)])
            
            constraint_name = f"Control_Square_{r}_{c}"
            prob += pulp.lpSum(controlling_squares) >= 1, constraint_name
            
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    except pulp.apis.PulpSolverError:
        print("Error: PuLP or its solver (CBC) is not installed correctly.")
        print("Please run: pip install pulp")
        return None
    
    min_knights = int(pulp.value(prob.objective))
    
    print("--- Knight's Dominating Set Solution ---")
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"Minimum Knights Required: {min_knights}")
    print("\nOne possible placement ('.'=empty, 'N'=Knight):")
    
    board = [["." for _ in range(board_size)] for _ in range(board_size)]
    
    for r in range(board_size):
        for c in range(board_size):
            if x[(r, c)].varValue == 1:
                board[r][c] = "N"

    print("  " + " ".join([str(i) for i in range(board_size)]))
    print(" +" + "--" * board_size)
    for i, row in enumerate(board):
        print(f"{i}| {' '.join(row)}")
    print("------------------------------------------")
    
    return min_knights

if __name__ == "__main__":
    solve_knight_problem()