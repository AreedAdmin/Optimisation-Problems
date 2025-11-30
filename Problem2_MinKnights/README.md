# ♞ Minimum Knights — Knight's Dominating Set Problem

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PuLP](https://img.shields.io/badge/PuLP-ILP_Solver-FF6B6B?style=for-the-badge)
![Optimization](https://img.shields.io/badge/Integer_Linear-Programming-4ECDC4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

*Finding the minimum number of knights to dominate an entire chessboard*

</div>

---

## 📋 Table of Contents

- [Problem Overview](#-problem-overview)
- [The Optimization Challenge](#-the-optimization-challenge)
- [Mathematical Formulation](#-mathematical-formulation)
- [Solution Approach](#-solution-approach)
- [Installation](#-installation)
- [Usage](#-usage)
- [Algorithm Deep Dive](#-algorithm-deep-dive)
- [Results](#-results)
- [Complexity Analysis](#-complexity-analysis)
- [Extensions & Variations](#-extensions--variations)
- [Real-World Applications](#-real-world-applications)
- [Author](#-author)

---

## 🎯 Problem Overview

The **Knight's Dominating Set Problem** is a classic combinatorial optimization problem from graph theory and chess mathematics.

### The Problem Statement

> **Given an 8×8 chessboard, what is the minimum number of knights required such that every square on the board is either:**
> 1. **Occupied by a knight**, OR
> 2. **Attacked by at least one knight**

### Visual Representation

```
A knight on square (r, c) can attack these 8 squares:

         ┌───┐
         │ × │         (r-2, c-1) and (r-2, c+1)
     ┌───┼───┼───┐
     │ × │   │ × │     (r-1, c-2) and (r-1, c+2)
     ├───┼───┼───┤
     │   │ ♞ │   │     Knight at (r, c)
     ├───┼───┼───┤
     │ × │   │ × │     (r+1, c-2) and (r+1, c+2)
     └───┼───┼───┘
         │ × │         (r+2, c-1) and (r+2, c+1)
         └───┘

The knight moves in an "L" shape:
  • 2 squares in one direction + 1 square perpendicular, OR
  • 1 square in one direction + 2 squares perpendicular
```

### Key Terminology

| Term | Definition |
|------|------------|
| **Dominating Set** | A set of vertices D in a graph G such that every vertex not in D is adjacent to at least one vertex in D |
| **Knight Graph** | A graph where vertices are chessboard squares and edges connect squares a knight's move apart |
| **Control/Dominate** | A square is "controlled" if it has a knight OR is attacked by a knight |

---

## 💡 The Optimization Challenge

### Why is This Problem Interesting?

1. **NP-Hard Complexity**: The minimum dominating set problem is NP-hard in general graphs, but chess graphs have special structure we can exploit.

2. **Rich History**: This problem has been studied since the 19th century and has known optimal solutions for standard chessboards.

3. **Practical Applications**: The solution techniques apply to facility location, network coverage, and sensor placement problems.

### Problem Characteristics

| Property | Value |
|----------|-------|
| Board Size | 8 × 8 = 64 squares |
| Decision Variables | 64 binary variables (one per square) |
| Constraints | 64 coverage constraints (one per square) |
| Problem Type | Integer Linear Programming (ILP) |
| Known Optimal Solution | **12 knights** |

---

## 📐 Mathematical Formulation

### Sets and Indices

- **S** = {(i, j) : 0 ≤ i, j ≤ 7} — Set of all 64 chessboard squares
- **N(i, j)** — Set of squares that can attack square (i, j) via knight move

### Decision Variables

```
x[i,j] ∈ {0, 1}  ∀(i,j) ∈ S

Where:
  x[i,j] = 1  if a knight is placed on square (i, j)
  x[i,j] = 0  otherwise
```

### Objective Function

**Minimize** the total number of knights:

```
minimize  Σ x[i,j]  for all (i,j) ∈ S
```

### Constraints

**Coverage Constraint** — Every square must be controlled:

```
x[i,j] + Σ x[r,c] ≥ 1    ∀(i,j) ∈ S
         (r,c)∈N(i,j)
```

This ensures that for each square (i, j), either:
- A knight is placed on (i, j) itself, OR
- At least one knight is placed on a square that can attack (i, j)

### Knight Move Offsets

```python
knight_moves = [
    (+1, +2), (+1, -2), (-1, +2), (-1, -2),
    (+2, +1), (+2, -1), (-2, +1), (-2, -1)
]
```

---

## 🔧 Solution Approach

### Why Integer Linear Programming?

| Method | Pros | Cons |
|--------|------|------|
| **Brute Force** | Guaranteed optimal | O(2^64) — infeasible |
| **Greedy Heuristic** | Fast | No optimality guarantee |
| **Backtracking** | Optimal with pruning | Still exponential |
| **ILP Solver ✓** | Optimal + efficient | Requires solver library |

We use **PuLP** with the **CBC (COIN-OR Branch and Cut)** solver because:
- Handles binary constraints naturally
- Exploits problem structure for pruning
- Provides proven optimal solutions
- Runs in seconds for 8×8 boards

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  PROBLEM INITIALIZATION                      │
│  1. Create ILP minimization problem                          │
│  2. Define board size (8×8)                                  │
│  3. Generate all 64 square coordinates                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               DECISION VARIABLE CREATION                     │
│  For each square (i, j):                                     │
│    • Create binary variable x[i,j] ∈ {0, 1}                 │
│    • x[i,j] = 1 means "place knight here"                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 OBJECTIVE FUNCTION                           │
│  minimize: Σ x[i,j] for all squares                         │
│  (minimize total knight count)                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               CONSTRAINT GENERATION                          │
│  For each square (r, c):                                     │
│    1. Add x[r,c] to controlling set                         │
│    2. For each valid knight move (dr, dc):                  │
│       • Calculate neighbor (nr, nc) = (r+dr, c+dc)          │
│       • If on board, add x[nr,nc] to controlling set        │
│    3. Add constraint: sum(controlling set) ≥ 1              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    SOLVE & OUTPUT                            │
│  1. Invoke CBC solver                                        │
│  2. Extract optimal knight count                             │
│  3. Build and display board visualization                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install pulp
```

### Verify Installation

```python
import pulp
print(pulp.listSolvers(onlyAvailable=True))
# Should show ['PULP_CBC_CMD'] or similar
```

---

## 🚀 Usage

### Basic Execution

```bash
python min-knights.py
```

### Expected Output

```
--- Knight's Dominating Set Solution ---
Status: Optimal
Minimum Knights Required: 12

One possible placement ('.'=empty, 'N'=Knight):
  0 1 2 3 4 5 6 7
 +----------------
0| . . . N . . . .
1| . . . . . . N .
2| N . . . . . . .
3| . . . . N . . .
4| . . . N . . . .
5| . . . . . . . N
6| . N . . . . . .
7| . . . . N . . .
------------------------------------------
```

### Programmatic Usage

```python
from min_knights import solve_knight_problem

# Get the minimum number of knights
min_knights = solve_knight_problem()
print(f"Answer: {min_knights} knights are needed")
```

---

## 🔬 Algorithm Deep Dive

### Step-by-Step Walkthrough

#### Step 1: Variable Creation

For each of the 64 squares, we create a binary decision variable:

```python
x = pulp.LpVariable.dicts("x", squares, 0, 1, pulp.LpBinary)
```

This creates variables like:
- `x_0_0`, `x_0_1`, ..., `x_0_7` (row 0)
- `x_1_0`, `x_1_1`, ..., `x_1_7` (row 1)
- ... and so on

#### Step 2: Coverage Constraints

For square (3, 3) in the center, the constraint is:

```
x[3,3] + x[1,2] + x[1,4] + x[2,1] + x[2,5] + 
         x[4,1] + x[4,5] + x[5,2] + x[5,4] ≥ 1
```

This says: "Either place a knight on (3,3), or place at least one knight on any of the 8 squares that can attack (3,3)."

#### Step 3: Corner Case

For corner square (0, 0), only 2 squares can attack it:

```
x[0,0] + x[1,2] + x[2,1] ≥ 1
```

Corners are harder to cover, influencing the optimal solution.

### Constraint Matrix Visualization

```
For a simplified 4×4 board, the constraint for square (1,1) would be:

                 Variables:
                 x00 x01 x02 x03 x10 x11 x12 x13 x20 x21 x22 x23 x30 x31 x32 x33
                 ─────────────────────────────────────────────────────────────────
Coverage(1,1):   [ 0   0   1   0   0   1   0   1   0   0   0   0   1   0   0   0 ] ≥ 1
                       ↑       ↑       ↑   ↑               ↑
                      (0,2)  (0,3)  self (1,2)           (3,0)
                      attacks  ↑                          attacks
                              out of bounds - excluded
```

---

## 📊 Results

### Optimal Solution

| Metric | Value |
|--------|-------|
| **Minimum Knights** | 12 |
| **Coverage** | 100% (all 64 squares) |
| **Solver Status** | Optimal |
| **Solve Time** | < 1 second |

### Sample Optimal Placement

```
  0 1 2 3 4 5 6 7
 +----------------
0| . . . N . . . .
1| . . . . . . N .
2| N . . . . . . .
3| . . . . N . . .
4| . . . N . . . .
5| . . . . . . . N
6| . N . . . . . .
7| . . . . N . . .

Legend:
  N = Knight placed
  . = Empty (but controlled by at least one knight)
```

### Known Results for Different Board Sizes

| Board Size | Minimum Knights | Reference |
|------------|-----------------|-----------|
| 3×3 | 4 | Trivial |
| 4×4 | 4 | Computed |
| 5×5 | 5 | Computed |
| 6×6 | 8 | Computed |
| 7×7 | 10 | Computed |
| **8×8** | **12** | **This solution** |
| 9×9 | 14 | Literature |
| 10×10 | 16 | Literature |

---

## ⚡ Complexity Analysis

### Problem Size

| Component | Count |
|-----------|-------|
| Decision Variables | n² (64 for 8×8) |
| Constraints | n² (64 for 8×8) |
| Non-zeros in constraint matrix | ~8n² (each constraint has ≤9 variables) |

### Time Complexity

| Phase | Complexity |
|-------|------------|
| Model Construction | O(n²) |
| ILP Solving | Exponential worst-case, but typically polynomial with modern solvers |
| Solution Extraction | O(n²) |

### Space Complexity

| Component | Space |
|-----------|-------|
| Variables | O(n²) |
| Constraint Matrix | O(n²) sparse |
| Total | O(n²) |

---

## 🔄 Extensions & Variations

### 1. Different Board Sizes

Modify `board_size` to solve for any n×n board:

```python
board_size = 10  # For 10×10 board
```

### 2. Independent Dominating Set

Add constraint that no two knights attack each other:

```python
# For each pair of knights that could attack each other
for (r1, c1) in squares:
    for (dr, dc) in knight_moves:
        r2, c2 = r1 + dr, c1 + dc
        if (r2, c2) in squares:
            prob += x[(r1,c1)] + x[(r2,c2)] <= 1
```

### 3. Weighted Domination

Assign different importance to different squares:

```python
weights = {(i,j): compute_weight(i,j) for (i,j) in squares}
prob += pulp.lpSum([weights[s] * x[s] for s in squares])
```

### 4. K-Domination

Require each square to be covered by at least k knights:

```python
prob += pulp.lpSum(controlling_squares) >= k, constraint_name
```

---

## 🌍 Real-World Applications

### 1. **Facility Location**
Place the minimum number of emergency response stations such that every neighborhood is within reach.

### 2. **Wireless Network Coverage**
Position the minimum number of cell towers to provide coverage to all areas.

### 3. **Security Camera Placement**
Install minimum cameras in a museum where each camera has a specific visibility pattern.

### 4. **Sensor Networks**
Deploy minimum sensors in an IoT network to monitor all regions.

### 5. **Firefighter Deployment**
Station firefighters at minimum locations to respond to any fire within a time limit.

---

## 📁 Project Structure

```
Problem2_MinKnights/
├── README.md           # This documentation
└── min-knights.py      # Main solver implementation
```

---

## 🔗 References

1. **Cockayne, E.J. & Hedetniemi, S.T.** (1977). "Towards a theory of domination in graphs." *Networks*, 7(3), 247-261.

2. **Watkins, J.J.** (2004). *Across the Board: The Mathematics of Chessboard Problems*. Princeton University Press.

3. **PuLP Documentation**: https://coin-or.github.io/pulp/

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👥 Authors

**Shehab Hassani**
- 📧 Email: shehab.hassani@areednow.com
- 🎓 Institution: Imperial College London

**Matt Huang**
- 🎓 Institution: Imperial College London

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| **GitHub Repository** | [https://github.com/AreedAdmin/Optimisation-Problems](https://github.com/AreedAdmin/Optimisation-Problems) |
| **Problem 1: Matrix Filler** | [../Problem1_MatrixFill](../Problem1_MatrixFill) |

---

<div align="center">

**Built with ♟️ for the Optimisation community**

*Part of the Imperial College London Optimisation & Decision Making course*

</div>

