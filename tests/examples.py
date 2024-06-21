import numpy as np

# Quadratic function definition
# Objective: minimize x^2 + y^2 + (z + 1)^2
# Gradient: [2*x, 2*y, 2*(z + 1)]
# Hessian: [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
def qp_function(x):
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    g = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    h = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])
    return f, g, h

# Inequality constraint: x >= 0
# Gradient: [-1, 0, 0]
# Hessian: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
def qp_inequality1(x):
    f = -x[0]
    g = np.array([-1, 0, 0])
    h = np.zeros((3, 3))
    return f, g, h

# Inequality constraint: y >= 0
# Gradient: [0, -1, 0]
# Hessian: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
def qp_inequality2(x):
    f = -x[1]
    g = np.array([0, -1, 0])
    h = np.zeros((3, 3))
    return f, g, h

# Inequality constraint: z >= 0
# Gradient: [0, 0, -1]
# Hessian: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
def qp_inequality3(x):
    f = -x[2]
    g = np.array([0, 0, -1])
    h = np.zeros((3, 3))
    return f, g, h

# The equality constraint of the quadratic function of x+y+z=1 is handled in test_constrained_min.

# Linear function definition
# Objective: maximize x + y, written as minimize -x - y
# This transformation does not change the optimal solution, just enables us to use a standard form in our solver
# Gradient: [-1, -1]
# Hessian: [[0, 0], [0, 0]]
def lp_function(x):
    f = -x[0] - x[1]
    g = np.array([-1, -1])
    h = np.zeros((2, 2))
    return f, g, h

# Inequality constraint: y <= 1
# Gradient: [0, 1]
# Hessian: [[0, 0], [0, 0]]
def lp_inequality1(x):
    f = x[1] - 1
    g = np.array([0, 1])
    h = np.zeros((2, 2))
    return f, g, h

# Inequality constraint: x <= 2
# Gradient: [1, 0]
# Hessian: [[0, 0], [0, 0]]
def lp_inequality2(x):
    f = x[0] - 2
    g = np.array([1, 0])
    h = np.zeros((2, 2))
    return f, g, h

# Inequality constraint: y >= 0
# Gradient: [0, -1]
# Hessian: [[0, 0], [0, 0]]
def lp_inequality3(x):
    f = -x[1]
    g = np.array([0, -1])
    h = np.zeros((2, 2))
    return f, g, h

# Inequality constraint: y >= -x + 1 <=> -x - y + 1 <= 0
# Gradient: [-1, -1]
# Hessian: [[0, 0], [0, 0]]
def lp_inequality4(x):
    f = -x[0] - x[1] + 1
    g = np.array([-1, -1])
    h = np.zeros((2, 2))
    return f, g, h