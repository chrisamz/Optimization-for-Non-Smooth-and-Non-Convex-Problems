# application_testing.py

"""
Application Testing Module for Optimization for Non-Smooth and Non-Convex Problems

This module contains functions for applying the optimization algorithms to real-world problems and testing their performance.

Use Cases:
- Energy Management
- Logistics Optimization
- Financial Portfolio Management

Libraries/Tools:
- numpy
- matplotlib
- pandas

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from optimization_algorithms import OptimizationAlgorithms
from model_evaluation import ModelEvaluation

class ApplicationTesting:
    def __init__(self, func, bounds, optimization_method, **kwargs):
        """
        Initialize the ApplicationTesting class.
        
        :param func: callable, objective function to be optimized
        :param bounds: list of tuples, bounds for each dimension of the function
        :param optimization_method: str, method to be used for optimization ('random_search', 'simulated_annealing', 'genetic_algorithm', 'pso')
        :param kwargs: additional parameters for the optimization method
        """
        self.func = func
        self.bounds = bounds
        self.optimization_method = optimization_method
        self.kwargs = kwargs
        self.optimizer = OptimizationAlgorithms(func, bounds)
        self.evaluator = ModelEvaluation()

    def optimize(self):
        """
        Perform optimization using the specified method.
        
        :return: dict, best solution and its value
        """
        start_time = time.time()
        
        if self.optimization_method == 'random_search':
            result = self.optimizer.random_search(**self.kwargs)
        elif self.optimization_method == 'simulated_annealing':
            result = self.optimizer.simulated_annealing()
        elif self.optimization_method == 'genetic_algorithm':
            result = self.optimizer.genetic_algorithm(**self.kwargs)
        elif self.optimization_method == 'pso':
            result = self.optimizer.particle_swarm_optimization(**self.kwargs)
        else:
            raise ValueError("Invalid optimization method. Choose 'random_search', 'simulated_annealing', 'genetic_algorithm', or 'pso'.")
        
        end_time = time.time()
        result['start_time'] = start_time
        result['end_time'] = end_time
        return result

    def evaluate(self, result, optimization_history, output_dir):
        """
        Evaluate the optimization results.
        
        :param result: dict, result of the optimization process
        :param optimization_history: list, history of objective values during optimization
        :param output_dir: str, directory to save the evaluation results
        """
        best_solution = result['solution']
        start_time = result['start_time']
        end_time = result['end_time']
        
        self.evaluator.evaluate(optimization_history, best_solution, self.func, start_time, end_time, output_dir)

if __name__ == "__main__":
    def energy_management_objective(x):
        # Example objective function for energy management
        return np.sum(np.square(x - np.mean(x)))

    def logistics_optimization_objective(x):
        # Example objective function for logistics optimization
        return np.sum(np.abs(x - np.median(x)))

    def financial_portfolio_objective(x):
        # Example objective function for financial portfolio management
        return -np.sum(np.log(x + 1))

    bounds = [(0, 10) for _ in range(10)]  # Example bounds for a 10-dimensional problem

    # Example use case: Energy Management
    energy_management_testing = ApplicationTesting(func=energy_management_objective, bounds=bounds, optimization_method='simulated_annealing')
    energy_management_result = energy_management_testing.optimize()
    energy_management_history = [energy_management_objective(np.random.uniform(0, 10, 10)) for _ in range(100)]
    energy_management_testing.evaluate(energy_management_result, energy_management_history, 'results/energy_management')

    # Example use case: Logistics Optimization
    logistics_optimization_testing = ApplicationTesting(func=logistics_optimization_objective, bounds=bounds, optimization_method='genetic_algorithm', n_generations=50, population_size=20)
    logistics_optimization_result = logistics_optimization_testing.optimize()
    logistics_optimization_history = [logistics_optimization_objective(np.random.uniform(0, 10, 10)) for _ in range(100)]
    logistics_optimization_testing.evaluate(logistics_optimization_result, logistics_optimization_history, 'results/logistics_optimization')

    # Example use case: Financial Portfolio Management
    financial_portfolio_testing = ApplicationTesting(func=financial_portfolio_objective, bounds=bounds, optimization_method='pso', swarmsize=50, maxiter=100)
    financial_portfolio_result = financial_portfolio_testing.optimize()
    financial_portfolio_history = [financial_portfolio_objective(np.random.uniform(0, 10, 10)) for _ in range(100)]
    financial_portfolio_testing.evaluate(financial_portfolio_result, financial_portfolio_history, 'results/financial_portfolio')

    print("Application testing completed and results saved.")
