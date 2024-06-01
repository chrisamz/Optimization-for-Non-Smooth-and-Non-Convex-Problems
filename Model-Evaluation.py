# model_evaluation.py

"""
Model Evaluation Module for Optimization for Non-Smooth and Non-Convex Problems

This module contains functions for evaluating the performance of the optimization algorithms.

Techniques Used:
- Convergence Rate Analysis
- Solution Quality Assessment
- Computational Efficiency Measurement

Libraries/Tools:
- numpy
- matplotlib

"""

import numpy as np
import matplotlib.pyplot as plt
import os

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def convergence_rate(self, optimization_history):
        """
        Calculate and plot the convergence rate of the optimization algorithm.
        
        :param optimization_history: list, history of objective values during optimization
        :return: float, final objective value
        """
        plt.figure(figsize=(10, 6))
        plt.plot(optimization_history, label='Objective Value')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Convergence Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

        final_value = optimization_history[-1]
        return final_value

    def solution_quality(self, best_solution, objective_function):
        """
        Assess the quality of the best solution found by the optimization algorithm.
        
        :param best_solution: array, best solution found by the optimization algorithm
        :param objective_function: callable, objective function to evaluate the solution
        :return: float, objective value of the best solution
        """
        objective_value = objective_function(best_solution)
        return objective_value

    def computational_efficiency(self, start_time, end_time):
        """
        Measure the computational efficiency of the optimization algorithm.
        
        :param start_time: float, start time of the optimization process
        :param end_time: float, end time of the optimization process
        :return: float, total computational time in seconds
        """
        total_time = end_time - start_time
        return total_time

    def evaluate(self, optimization_history, best_solution, objective_function, start_time, end_time, output_dir):
        """
        Execute the full evaluation pipeline.
        
        :param optimization_history: list, history of objective values during optimization
        :param best_solution: array, best solution found by the optimization algorithm
        :param objective_function: callable, objective function to evaluate the solution
        :param start_time: float, start time of the optimization process
        :param end_time: float, end time of the optimization process
        :param output_dir: str, directory to save the evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Calculate convergence rate
        final_value = self.convergence_rate(optimization_history)
        print(f"Final Objective Value: {final_value}")

        # Assess solution quality
        objective_value = self.solution_quality(best_solution, objective_function)
        print(f"Objective Value of Best Solution: {objective_value}")

        # Measure computational efficiency
        total_time = self.computational_efficiency(start_time, end_time)
        print(f"Total Computational Time: {total_time} seconds")

        # Save evaluation results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Final Objective Value: {final_value}\n")
            f.write(f"Objective Value of Best Solution: {objective_value}\n")
            f.write(f"Total Computational Time: {total_time} seconds\n")

        print(f"Evaluation results saved to {os.path.join(output_dir, 'evaluation_results.txt')}")

if __name__ == "__main__":
    import time

    def objective_function(x):
        # Example objective function: Rastrigin function
        return 10 * len(x) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

    # Example optimization history and best solution
    optimization_history = [objective_function(np.random.uniform(-5.12, 5.12, 10)) for _ in range(100)]
    best_solution = np.random.uniform(-5.12, 5.12, 10)

    # Example start and end time
    start_time = time.time()
    time.sleep(1)  # Simulate computation time
    end_time = time.time()

    output_dir = 'results/evaluation/'
    evaluator = ModelEvaluation()

    # Evaluate the optimization results
    evaluator.evaluate(optimization_history, best_solution, objective_function, start_time, end_time, output_dir)
    print("Model evaluation completed and results saved.")
