# imports and set up
import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import plot_qp_path, plot_lp_path, plot_objective_values
from src.constrained_min import interior_pt
from tests.examples import qp_function, lp_function, qp_inequality1, qp_inequality2, qp_inequality3, lp_inequality1, lp_inequality2, lp_inequality3, lp_inequality4

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.starting_points = {
            "quadratic_program": np.array([0.1, 0.2, 0.7]),
            "linear_program": np.array([0.5, 0.75])
        }

    def execute_minimization(self, function, inequality_constraints, equality_matrix, equality_rhs, initial_point):
        optimal_point, optimal_value, inequality_values_at_optimal, path, obj_values = interior_pt(
        function, inequality_constraints, equality_matrix, equality_rhs, initial_point)
        inequality_values_at_optimal = [constraint(optimal_point)[0] for constraint in inequality_constraints]
    
        if equality_matrix is not None:
            equality_values_at_optimal = np.dot(equality_matrix, optimal_point)
        else:
            equality_values_at_optimal = None

        return optimal_point, optimal_value, inequality_values_at_optimal, equality_values_at_optimal, path, obj_values
    
    def test_qp(self):
        case_name = "Quadratic Programming Test"
        function = qp_function
        inequality_constraints = [qp_inequality1, qp_inequality2, qp_inequality3]
        equality_matrix = np.array([1, 1, 1]).reshape(1, 3)
        equality_rhs = np.array([1])  # Right-hand side of the equality constraint
        initial_point = self.starting_points["quadratic_program"]

        optimal_point, optimal_value, inequality_values_at_optimal, equality_values_at_optimal, path, obj_values = self.execute_minimization(
        function, inequality_constraints, equality_matrix, equality_rhs, initial_point)
        print(f"----- Test Case: {case_name} Summary -----")
        print(f"Optimal point - final candidate: {[f'{val:.10f}' for val in optimal_point]}")
        print(f"Optimal objective value: {optimal_value:.10f}")
        print(f"Inequality constraint values at optimal point: {[f'{val:.10f}' for val in inequality_values_at_optimal]}")
        if equality_values_at_optimal is not None:
            print(f"Equality constraint values at optimal point: {[f'{val:.10f}' for val in equality_values_at_optimal]}")

        plot_objective_values(obj_values, 'Objective values per iteration - Quadratic function')
        plot_qp_path(path, 'Path and feasible region - Quadratic function')
    
    def test_lp(self):
        case_name = "Linear Programming Test"
        function = lp_function
        inequality_constraints = [lp_inequality1, lp_inequality2, lp_inequality3, lp_inequality4]
        equality_matrix = None
        initial_point = self.starting_points["linear_program"]

        optimal_point, optimal_value, inequality_values_at_optimal, equality_values_at_optimal, path, obj_values = self.execute_minimization(function, inequality_constraints, equality_matrix, 0, initial_point)
        print(f"----- Test Case: {case_name} Summary -----")
        print(f"Optimal point - final candidate: {[f'{val:.10f}' for val in optimal_point]}")
        print(f"Optimal objective value: {optimal_value:.10f}")
        print(f"Inequality constraint values at optimal point: {[f'{val:.10f}' for val in inequality_values_at_optimal]}")
        if equality_values_at_optimal is not None:
            print(f"Equality constraint values at optimal point: {[f'{val:.10f}' for val in equality_values_at_optimal]}")

        plot_objective_values(obj_values, 'Objective values per iteration - Linear function')
        plot_lp_path(path, 'Path and feasible region - Linear function')

if __name__ == '__main__':
    unittest.main()