# 🧮 Matrix Filler — Constrained Matrix Completion Solver

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![PyPI](https://img.shields.io/badge/PyPI-Published-brightgreen?style=for-the-badge&logo=pypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

*A robust optimization solver for constrained matrix completion problems*

**Published on PyPI:** `pip install matrix-filler`

</div>

---

## 📋 Table of Contents

- [Problem Overview](#-problem-overview)
- [The Optimization Challenge](#-the-optimization-challenge)
- [Mathematical Formulation](#-mathematical-formulation)
- [Solution Approach](#-solution-approach)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [Algorithm Deep Dive](#-algorithm-deep-dive)
- [Performance Considerations](#-performance-considerations)
- [Real-World Applications](#-real-world-applications)
- [Project Structure](#-project-structure)
- [Manual Solution Reference](#-manual-solution-reference)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Problem Overview

This project tackles a classic **constrained matrix completion problem** encountered in operations research and optimization courses at Imperial College London.

### The Problem Statement

Given a partially filled matrix where:
- **Some cells contain pre-filled (known) values** that must remain unchanged
- **Some cells are empty (unknown)** and need to be determined
- **Each row has a target sum** that must be achieved
- **Each column has a target sum** that must be achieved

**Objective:** Find values for all empty cells such that the row and column constraints are satisfied as closely as possible.

### Visual Representation

```
                    Column Targets
                    ↓    ↓    ↓    ↓
                   10   10   10   10
                   ─────────────────
              10 → │ 2 │ ? │ ? │ 3 │
Row Targets   10 → │ ? │ 1 │ ? │ ? │
              10 → │ ? │ ? │ 4 │ ? │
                   ─────────────────

Where:
  • Numbers (2, 3, 1, 4) = Pre-filled values (FIXED)
  • ? = Unknown values (TO BE SOLVED)
  • Row/Column arrows = Target sums (CONSTRAINTS)
```

### Challenge Characteristics

1. **Over-constrained or Under-constrained Systems**: The number of unknowns may not match the number of equations
2. **Non-square Matrices**: The solver must handle matrices of any dimension (m × n)
3. **Continuous Values**: Solutions can be any real number (not restricted to integers)
4. **Bound Constraints**: Optional enforcement of non-negativity
5. **Feasibility**: Not all constraint combinations have exact solutions

---

## 💡 The Optimization Challenge

### Why is This Problem Non-Trivial?

At first glance, this might seem like a simple system of linear equations. However, several factors make it genuinely challenging:

#### 1. **System Overdetermination**

For an `m × n` matrix with `k` unknown cells:
- We have `m + n` constraint equations (one per row + one per column)
- We have `k` unknown variables

Typically, `m + n ≠ k`, meaning:
- **Overdetermined** (`m + n > k`): More equations than unknowns → Exact solution may not exist
- **Underdetermined** (`m + n < k`): Fewer equations than unknowns → Infinitely many solutions

#### 2. **Constraint Consistency**

The row targets and column targets must be consistent:
```
Sum of all row targets = Sum of all column targets = Grand total of matrix
```
If this invariant is violated, no exact solution exists.

#### 3. **Pre-filled Values Impact**

Existing values reduce the degrees of freedom and may create conflicts with the targets.

#### 4. **Non-negativity Requirements**

In many practical applications (e.g., quantities, counts, costs), negative values are meaningless, adding inequality constraints to the problem.

---

## 📐 Mathematical Formulation

### Problem Definition

Let:
- **G** ∈ ℝ^(m×n) be the input matrix with some entries known and others unknown (NaN)
- **r** ∈ ℝ^m be the vector of row target sums
- **c** ∈ ℝ^n be the vector of column target sums
- **X** = {x₁, x₂, ..., xₖ} be the set of k unknown cell values
- **(iⱼ, jⱼ)** denote the position of the j-th unknown

### Constraint Formulation

**Row Constraints:**
For each row i ∈ {1, ..., m}:
```
∑(known values in row i) + ∑(unknowns in row i) = rᵢ
```

**Column Constraints:**
For each column j ∈ {1, ..., n}:
```
∑(known values in column j) + ∑(unknowns in column j) = cⱼ
```

### Linear System Representation

This can be written as a linear system **Ax = b** where:

```
A ∈ ℝ^((m+n) × k)  — Constraint coefficient matrix
x ∈ ℝ^k            — Vector of unknown values
b ∈ ℝ^(m+n)        — Adjusted target vector
```

Where:
- **A[i, j] = 1** if unknown xⱼ appears in the i-th constraint
- **b[i] = target[i] - sum(known values in constraint i)**

### Optimization Objective

Since exact solutions may not exist, we minimize the **sum of squared residuals**:

```
minimize  ‖Ax - b‖₂²

subject to:
            xⱼ ≥ 0  ∀j ∈ {1,...,k}  (if non_negative=True)
```

This is a **Bounded Least Squares (BLS)** problem.

---

## 🔧 Solution Approach

### Why Bounded Least Squares?

We chose **scipy.optimize.lsq_linear** for several compelling reasons:

| Approach | Pros | Cons |
|----------|------|------|
| **Direct Solve (np.linalg.solve)** | Fast, exact | Only works for square, non-singular systems |
| **Pseudo-inverse (np.linalg.lstsq)** | Handles overdetermined | Cannot enforce bounds |
| **General LP/QP Solvers** | Full constraint support | Slower, complex setup |
| **Bounded Least Squares ✓** | Fast, handles bounds, robust | Approximate solution |

### Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PROCESSING                          │
│  1. Parse input matrix G, identify NaN positions            │
│  2. Map unknowns to variable indices: (i,j) → k             │
│  3. Count: k unknowns, m rows, n columns                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 CONSTRAINT MATRIX CONSTRUCTION               │
│  For each row r:                                             │
│    • Initialize b[r] = row_target[r]                        │
│    • For each cell (r, c):                                  │
│        - If known: b[r] -= value                            │
│        - If unknown: A[r, k] = 1                            │
│                                                              │
│  For each column c:                                          │
│    • Initialize b[m+c] = col_target[c]                      │
│    • For each cell (r, c):                                  │
│        - If known: b[m+c] -= value                          │
│        - If unknown: A[m+c, k] = 1                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    BOUNDS CONFIGURATION                      │
│  If non_negative=True:                                       │
│    lower_bounds = [0, 0, ..., 0]      (k zeros)             │
│  Else:                                                       │
│    lower_bounds = [-∞, -∞, ..., -∞]   (k negative infinities)│
│                                                              │
│  upper_bounds = [+∞, +∞, ..., +∞]     (always unbounded above)│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                SCIPY LSQ_LINEAR OPTIMIZATION                 │
│  result = lsq_linear(A, b, bounds=(lb, ub))                 │
│                                                              │
│  Internally uses:                                            │
│    • BVLS (Bounded Variable Least Squares) algorithm        │
│    • Active set method for bound constraints                │
│    • QR factorization for numerical stability               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    SOLUTION RECONSTRUCTION                   │
│  For each unknown k at position (i, j):                     │
│    filled_grid[i, j] = result.x[k]                          │
│                                                              │
│  Return: (filled_grid, optimization_result)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔲 Universal Matrix Support

```python
# Square matrices
3×3, 4×4, 5×5, ..., n×n  ✓

# Non-square matrices  
3×5, 4×2, 10×3, m×n  ✓

# Any sparsity pattern
Fully empty, mostly filled, diagonal, random  ✓
```

### 📊 Flexible Value Types

```python
# Continuous (real) values
[2.5, 3.14159, -7.8, 0.001]  ✓

# Integer values (solved as continuous)
[1, 2, 3, 4, 5]  ✓

# Mixed pre-filled patterns
[[1, nan, 3], [nan, nan, nan], [7, 8, nan]]  ✓
```

### ⚙️ Configurable Constraints

```python
# Non-negative only (default)
non_negative=True  → x ≥ 0

# Allow negative values
non_negative=False → x ∈ ℝ
```

### 📈 Rich Diagnostic Output

The solver returns comprehensive optimization diagnostics:

```python
result.success    # Boolean: Did optimization succeed?
result.message    # String: Detailed status message
result.x          # Array: Solved unknown values
result.cost       # Float: Final objective value ‖Ax-b‖²
result.optimality # Float: Optimality measure
result.nit        # Int: Number of iterations
```

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install matrix-filler
```

### From Source

```bash
git clone https://github.com/AreedAdmin/matrix-filler.git
cd matrix-filler
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| NumPy | Latest | Array operations |
| SciPy | Latest | Optimization engine |

---

## 📖 Usage Guide

### Basic Example

```python
import numpy as np
from matrix_filler import fill_matrix_with_constraints

# Define a matrix with unknown values (np.nan)
grid = np.array([
    [2.0, np.nan, np.nan, 3.0],
    [np.nan, 1.0, np.nan, np.nan],
    [np.nan, np.nan, 4.0, np.nan],
    [3.0, np.nan, np.nan, 1.0]
])

# Define target sums
row_targets = np.array([10, 10, 10, 10])
col_targets = np.array([10, 10, 10, 10])

# Solve!
filled_grid, result = fill_matrix_with_constraints(
    grid, 
    row_targets, 
    col_targets,
    non_negative=True
)

print("Filled Matrix:")
print(filled_grid)
print(f"\nSuccess: {result.success}")
print(f"Message: {result.message}")
```

### Non-Square Matrix Example

```python
import numpy as np
from matrix_filler import fill_matrix_with_constraints

# 3 rows × 4 columns (non-square)
grid = np.array([
    [2, np.nan, np.nan, 3.0],
    [np.nan, 1, np.nan, np.nan],
    [np.nan, np.nan, 4, np.nan]
])

row_targets = np.array([10, 10, 10])      # 3 targets
col_targets = np.array([10, 10, 10, 10])  # 4 targets

filled_grid, result = fill_matrix_with_constraints(
    grid, 
    row_targets, 
    col_targets,
    non_negative=True
)

print("Filled Matrix:")
print(filled_grid)

# Verify constraints
print(f"\nRow sums: {filled_grid.sum(axis=1)}")
print(f"Column sums: {filled_grid.sum(axis=0)}")
```

**Output:**
```
Filled Matrix:
[[2.         4.41428571 2.01428571 3.        ]
 [4.18571429 1.         2.55714286 3.68571429]
 [2.38571429 3.15714286 4.         1.88571429]]

Row sums: [11.42857143 11.42857143 11.42857143]
Column sums: [ 8.57142857  8.57142857  8.57142857 8.57142857]
```

### Allowing Negative Values

```python
import numpy as np
from matrix_filler import fill_matrix_with_constraints

grid = np.array([
    [10.0, np.nan, np.nan],
    [np.nan, 20.0, np.nan],
    [np.nan, np.nan, 30.0]
])

row_targets = np.array([15.0, 25.0, 35.0])
col_targets = np.array([20.0, 30.0, 25.0])

# Allow negative values
filled_grid, result = fill_matrix_with_constraints(
    grid, 
    row_targets, 
    col_targets, 
    non_negative=False  # Allows negative solutions
)

print("Filled Matrix:")
print(filled_grid)
```

### Validation and Diagnostics

```python
import numpy as np
from matrix_filler import fill_matrix_with_constraints

grid = np.array([
    [1.0, np.nan, 3.0],
    [np.nan, 2.0, np.nan],
    [4.0, np.nan, 5.0]
])

row_targets = np.array([10.0, 15.0, 20.0])
col_targets = np.array([12.0, 8.0, 25.0])

filled_grid, result = fill_matrix_with_constraints(grid, row_targets, col_targets)

# Comprehensive validation
print("=" * 50)
print("OPTIMIZATION RESULTS")
print("=" * 50)
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Cost (residual): {result.cost:.6f}")
print(f"Iterations: {result.nit}")
print(f"Optimality: {result.optimality:.2e}")

print("\n" + "=" * 50)
print("CONSTRAINT VERIFICATION")
print("=" * 50)
print(f"Target row sums:    {row_targets}")
print(f"Actual row sums:    {filled_grid.sum(axis=1)}")
print(f"Row error:          {np.abs(filled_grid.sum(axis=1) - row_targets).sum():.6f}")

print(f"\nTarget column sums: {col_targets}")
print(f"Actual column sums: {filled_grid.sum(axis=0)}")
print(f"Column error:       {np.abs(filled_grid.sum(axis=0) - col_targets).sum():.6f}")
```

---

## 📚 API Reference

### `fill_matrix_with_constraints(grid, row_targets, col_targets, non_negative=True)`

Fill `np.nan` entries in a 2D numpy array so that row and column sums match given targets, using bounded least squares optimization.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `grid` | `np.ndarray` | ✅ Yes | — | 2D array with known values and `np.nan` for unknowns |
| `row_targets` | `np.ndarray` | ✅ Yes | — | Desired sum of each row (length = number of rows) |
| `col_targets` | `np.ndarray` | ✅ Yes | — | Desired sum of each column (length = number of columns) |
| `non_negative` | `bool` | ❌ No | `True` | If `True`, forces all filled values to be ≥ 0 |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `filled_grid` | `np.ndarray` | Copy of input grid with all `np.nan` values filled |
| `result` | `scipy.optimize.OptimizeResult` | Optimization result object (see below) |

#### Result Object Properties

| Property | Type | Description |
|----------|------|-------------|
| `x` | `np.ndarray` | Solved values for all unknowns |
| `success` | `bool` | `True` if optimization converged successfully |
| `message` | `str` | Human-readable status message |
| `cost` | `float` | Final value of the objective function ‖Ax - b‖² |
| `nit` | `int` | Number of iterations performed |
| `optimality` | `float` | Measure of optimality (smaller = better) |
| `active_mask` | `np.ndarray` | Which bounds are active at the solution |

---

## 🔬 Algorithm Deep Dive

### Step-by-Step Walkthrough

Let's trace through a concrete example:

#### Input
```python
grid = np.array([
    [2, np.nan, 3],
    [np.nan, 1, np.nan]
])
row_targets = [10, 10]
col_targets = [6, 6, 8]
```

#### Step 1: Identify Unknowns

```
Position mapping:
  (0, 1) → x₀
  (1, 0) → x₁  
  (1, 2) → x₂

k = 3 unknowns
```

#### Step 2: Build Constraint Matrix A

```
Row 0: 2 + x₀ + 3 = 10  →  x₀ = 5        →  A[0] = [1, 0, 0]
Row 1: x₁ + 1 + x₂ = 10  →  x₁ + x₂ = 9  →  A[1] = [0, 1, 1]
Col 0: 2 + x₁ = 6        →  x₁ = 4        →  A[2] = [0, 1, 0]
Col 1: x₀ + 1 = 6        →  x₀ = 5        →  A[3] = [1, 0, 0]
Col 2: 3 + x₂ = 8        →  x₂ = 5        →  A[4] = [0, 0, 1]

     ┌─────────────┐      ┌───┐
     │ 1   0   0   │      │ 5 │
     │ 0   1   1   │      │ 9 │
A =  │ 0   1   0   │  b = │ 4 │
     │ 1   0   0   │      │ 5 │
     │ 0   0   1   │      │ 5 │
     └─────────────┘      └───┘
```

#### Step 3: Solve Ax = b

Using bounded least squares with lower bounds [0, 0, 0]:

```
x* = [5.0, 4.0, 5.0]
```

#### Step 4: Reconstruct Matrix

```
filled_grid = [
    [2, 5, 3],
    [4, 1, 5]
]
```

#### Verification

```
Row sums: [10, 10] ✓
Col sums: [6, 6, 8] ✓
```

---

## ⚡ Performance Considerations

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Unknown identification | O(mn) |
| Matrix A construction | O((m+n) × k) |
| BLS optimization | O(k³) worst case |
| Solution reconstruction | O(k) |

**Overall:** O(mn + k³) where k = number of unknowns

### Space Complexity

| Structure | Size |
|-----------|------|
| Constraint matrix A | (m + n) × k |
| Target vector b | m + n |
| Solution vector x | k |

**Overall:** O((m + n) × k)

### Practical Guidelines

| Matrix Size | Unknowns | Expected Time |
|-------------|----------|---------------|
| 10 × 10 | ~50 | < 1 ms |
| 100 × 100 | ~5,000 | ~10 ms |
| 1000 × 1000 | ~500,000 | ~seconds |

---

## 🌍 Real-World Applications

### 1. **Transportation Problems**
Determine shipment quantities between origins and destinations given supply and demand constraints.

### 2. **Survey Data Reconstruction**
Fill missing entries in contingency tables when only marginal totals are known.

### 3. **Economic Input-Output Analysis**
Complete inter-industry transaction tables with known row/column totals.

### 4. **Image Processing**
Reconstruct images from partial observations with intensity sum constraints.

### 5. **Financial Planning**
Allocate budgets across categories given departmental and project totals.

### 6. **Scheduling Problems**
Assign resources to time slots with capacity and demand constraints.

---

## 📁 Project Structure

```
Problem1_MatrixFill/
├── README.md                 # This documentation file
├── filler.py                 # Core solver implementation
├── example.ipynb             # Jupyter notebook with usage examples
├── manual_solution.pdf       # Hand-calculated solution reference
├── problem1_solution.pdf     # Detailed problem solution document
├── Prize 1.pdf               # Original competition/assignment brief
└── .gitignore                # Git ignore rules
```

### Related: Published Package Structure

```
matrix-filler/                # PyPI package
├── matrix_filler/
│   ├── __init__.py           # Package exports
│   └── filler.py             # Core implementation
├── pyproject.toml            # Package metadata
├── README.md                 # Package documentation
└── dist/                     # Built distributions
    ├── matrix_filler-0.1.0.tar.gz
    └── matrix_filler-0.1.0-py3-none-any.whl
```

---

## 📝 Manual Solution Reference

The `manual_solution.pdf` file contains hand-worked solutions demonstrating:

1. **Problem setup** — Translating the matrix problem to linear algebra
2. **Constraint derivation** — Building the equation system step by step
3. **Solution process** — Manual calculation of unknowns
4. **Verification** — Checking that constraints are satisfied

This serves as educational material to understand the underlying mathematics before using the automated solver.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- 🐛 **Bug Reports**: Open an issue describing the problem
- 💡 **Feature Requests**: Suggest new capabilities
- 📖 **Documentation**: Improve explanations or add examples
- 🔧 **Code**: Submit pull requests with improvements

### Development Setup

```bash
# Clone the repository
git clone https://github.com/AreedAdmin/matrix-filler.git
cd matrix-filler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests (if available)
pytest
```

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 Shehab Hassani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👤 Author

**Shehab Hassani**
**Matt Huang**

- 📧 Email: shehab.hassani@areednow.com
- 🎓 Institution: Imperial College London
- 📚 Course: Optimisation & Decision Models

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| **PyPI Package** | [https://pypi.org/project/matrix-filler/](https://pypi.org/project/matrix-filler/) |
| **GitHub Repository** | [https://github.com/AreedAdmin/matrix-filler](https://github.com/AreedAdmin/matrix-filler) |
| **Documentation** | This README |

---

<div align="center">

*If you find this project useful, consider giving it a ⭐ on GitHub!*

</div>

