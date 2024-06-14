#imports
import numpy as np
import math

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1
    mu = 10
    epsilon = 1e-8
    max_iter = 20 # Maximum iterations to prevent infinite loops
    x = x0.copy()
    path = [x0.copy()]
    obj_values = [func(x0)[0]]

    while (len(ineq_constraints) / t) > epsilon:
        for _ in range(max_iter):
            val, grad, hess = compute_combined_log_barrier_values(func, ineq_constraints, x, t)
            
            if eq_constraints_mat is not None:
                direction, _ = solve_kkt_system(hess, grad, eq_constraints_mat, eq_constraints_rhs - np.dot(eq_constraints_mat, x))
            else:
                direction = -np.linalg.solve(hess, grad)
            
            step_length = wolfe_condition_with_backtracking(lambda x: compute_combined_log_barrier_values(func, ineq_constraints, x, t), x, val, grad, direction)
            x = x + step_length * direction
            
            # Newton decrement check
            lambda_sq = np.dot(direction, np.dot(hess, direction))
            if 0.5 * lambda_sq < epsilon:
                break
        
        path.append(x.copy())
        obj_values.append(func(x)[0])
        t *= mu

    final_candidate = x
    final_val = func(final_candidate)[0]
    constraint_vals = [constraint(final_candidate)[0] for constraint in ineq_constraints]
    
    return final_candidate, final_val, constraint_vals, np.array(path), obj_values

def solve_kkt_system(hess, grad, A, b):
    # Solves the KKT system to handle equality constraints.
    KKT_matrix = np.block([
        [hess, A.T],
        [A, np.zeros((A.shape[0], A.shape[0]))]
    ])
    # Constructs the right-hand side of the KKT system.
    rhs = np.block([-grad, b])
    solution = np.linalg.solve(KKT_matrix, rhs)
    # Return the primal and dual directions.
    return solution[:grad.shape[0]], solution[grad.shape[0]:]

def wolfe_condition_with_backtracking(f, x, val, gradient, direction, alpha=0.01, beta=0.5, max_iter=20):
    step_length = 1
    curr_val = f(x + step_length * direction)[0]

    for _ in range(max_iter):
        if curr_val <= val + alpha * step_length * np.dot(gradient, direction):
            break
        step_length *= beta
        curr_val = f(x + step_length * direction)[0]
        
    return step_length

def compute_combined_log_barrier_values(f, ineq_constraints, x0, t):
    """
    Computes the combined objective value, gradient, and Hessian of the 
    objective function and the log barrier term for inequality constraints.

    Inputs:
        f: Function that returns the objective value, gradient, and Hessian at a given point.
        ineq_constraints: List of inequality constraint functions, each returning the 
                          constraint value, gradient, and Hessian at a given point.
        x0: Current point in the parameter space.
        t: Scalar multiplier for the objective function.

    Outputs:
        combined_val: Combined objective value and log barrier term.
        combined_grad: Combined gradient of the objective and log barrier term.
        combined_hess: Combined Hessian of the objective and log barrier term.
    """
    val, grad, hess = f(x0)
    x0_dim = x0.shape[0]
    log_f = 0
    log_g = np.zeros((x0_dim,))
    log_h = np.zeros((x0_dim, x0_dim))

    for constraint in ineq_constraints:
        f_c, g, h = constraint(x0)
        if f_c >= 0:
            return np.inf, np.zeros_like(x0), np.zeros((x0_dim, x0_dim))

        # Log barrier
        log_f += math.log(-f_c)
        log_g -= g / f_c

        grad_outer = np.outer(g, g) / (f_c ** 2)
        log_h += (h * f_c - grad_outer) / (f_c ** 2)

    combined_val = t * val - log_f
    combined_grad = t * grad - log_g
    combined_hess = t * hess - log_h

    return combined_val, combined_grad, combined_hess