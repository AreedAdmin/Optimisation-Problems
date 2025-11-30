import numpy as np
from scipy.optimize import lsq_linear

def fill_matrix_with_constraints(grid, row_targets, col_targets, non_negative=True):
    """
    Fill np.nan entries in a 2D numpy array so that row and column sums
    match given targets, using bounded least squares.

    Parameters
    ----------
    grid : np.ndarray
        2D array with known values and np.nan for unknowns.
    row_targets : np.ndarray
        Desired sum of each row (length = n_rows).
    col_targets : np.ndarray
        Desired sum of each column (length = n_cols).
    non_negative : bool
        If True, forces all filled values to be >= 0.

    Returns
    -------
    filled_grid : np.ndarray
        Copy of grid with all np.nan filled.
    result : scipy.optimize._lsq.lsq_linear.Result
        SciPy optimization result, including:
        - x (solved unknowns),
        - success (bool),
        - message (str).
    """
    grid = np.array(grid, dtype=float)
    row_targets = np.array(row_targets, dtype=float)
    col_targets = np.array(col_targets, dtype=float)

    n_rows, n_cols = grid.shape

    # 1. gather unknown positions
    unknown_positions = [(i, j) for i in range(n_rows) for j in range(n_cols) if np.isnan(grid[i, j])]
    num_unknowns = len(unknown_positions)

    # 2. build linear system A x = b
    num_equations = n_rows + n_cols
    A = np.zeros((num_equations, num_unknowns))
    b = np.zeros(num_equations)

    # row constraints
    for r in range(n_rows):
        b[r] = row_targets[r]
        for c in range(n_cols):
            val = grid[r, c]

            if np.isnan(val):
                idx = unknown_positions.index((r, c))
                A[r, idx] += 1.0
            else:
                b[r] -= val

    # column constraints
    for c in range(n_cols):
        eq_idx = n_rows + c
        b[eq_idx] = col_targets[c]

        for r in range(n_rows):
            val = grid[r, c]
            if np.isnan(val):
                idx = unknown_positions.index((r, c))
                A[eq_idx, idx] += 1.0
            else:
                b[eq_idx] -= val

    # 3. solve bounded least squares
    if non_negative:
        lower_bounds = np.zeros(num_unknowns)
    else:
        lower_bounds = -np.inf * np.ones(num_unknowns)

    upper_bounds = np.inf * np.ones(num_unknowns)

    result = lsq_linear(A, b, bounds=(lower_bounds, upper_bounds))

    # 4. fill grid with the solved values
    filled_grid = grid.copy()
    for k, (i, j) in enumerate(unknown_positions):
        filled_grid[i, j] = result.x[k]

    return filled_grid, result