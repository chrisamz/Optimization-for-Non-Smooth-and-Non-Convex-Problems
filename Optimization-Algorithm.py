# optimization_algorithms.py

"""
Optimization Algorithms Module for Optimization for Non-Smooth and Non-Convex Problems

This module contains functions for implementing zero-order nonsmooth nonconvex stochastic optimization algorithms.

Techniques Used:
- Random Search
- Simulated Annealing
- Genetic Algorithms
- Particle Swarm Optimization

Libraries/Tools:
- numpy
- scipy
- deap (for genetic algorithms)
- pyswarm (for particle swarm optimization)

"""

import numpy as np
from scipy.optimize import dual_annealing
from deap import base, creator, tools, algorithms
from pyswarm import pso
import random
import os
import joblib

class OptimizationAlgorithms:
    def __init__(self, func, bounds):
        """
        Initialize the OptimizationAlgorithms class.
        
        :param func: callable, objective function to be optimized
        :param bounds: list of tuples, bounds for each dimension of the function
        """
        self.func = func
        self.bounds = bounds

    def random_search(self, n_iter=1000):
        """
        Perform random search optimization.
        
        :param n_iter: int, number of iterations
        :return: dict, best solution and its value
        """
        best_solution = None
        best_value = float('inf')
        
        for _ in range(n_iter):
            candidate = np.array([random.uniform(low, high) for low, high in self.bounds])
            value = self.func(candidate)
            if value < best_value:
                best_value = value
                best_solution = candidate
        
        return {'solution': best_solution, 'value': best_value}

    def simulated_annealing(self):
        """
        Perform simulated annealing optimization.
        
        :return: dict, best solution and its value
        """
        result = dual_annealing(self.func, self.bounds)
        return {'solution': result.x, 'value': result.fun}

    def genetic_algorithm(self, n_generations=100, population_size=50, crossover_prob=0.7, mutation_prob=0.2):
        """
        Perform genetic algorithm optimization.
        
        :param n_generations: int, number of generations
        :param population_size: int, size of the population
        :param crossover_prob: float, probability of crossover
        :param mutation_prob: float, probability of mutation
        :return: dict, best solution and its value
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(self.bounds))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in self.bounds], up=[b[1] for b in self.bounds], eta=1.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", lambda ind: (self.func(np.array(ind)),))

        population = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

        best_solution = hof[0]
        best_value = self.func(np.array(best_solution))
        return {'solution': best_solution, 'value': best_value}

    def particle_swarm_optimization(self, swarmsize=100, maxiter=200):
        """
        Perform particle swarm optimization.
        
        :param swarmsize: int, number of particles in the swarm
        :param maxiter: int, maximum number of iterations
        :return: dict, best solution and its value
        """
        lb = [b[0] for b in self.bounds]
        ub = [b[1] for b in self.bounds]

        best_solution, best_value = pso(self.func, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
        return {'solution': best_solution, 'value': best_value}

    def save_model(self, model_dir, model_name='optimization_model.pkl'):
        """
        Save the optimization results to a file.
        
        :param model_dir: str, directory to save the model
        :param model_name: str, name of the model file
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        joblib.dump(self, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_dir, model_name='optimization_model.pkl'):
        """
        Load the optimization results from a file.
        
        :param model_dir: str, directory containing the model
        :param model_name: str, name of the model file
        :return: loaded model
        """
        model_path = os.path.join(model_dir, model_name)
        model = joblib.load(model_path)
        return model

if __name__ == "__main__":
    def objective_function(x):
        # Example objective function: Rastrigin function
        return 10 * len(x) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

    bounds = [(-5.12, 5.12) for _ in range(10)]  # Example bounds for a 10-dimensional problem
    optimizer = OptimizationAlgorithms(func=objective_function, bounds=bounds)

    # Perform random search optimization
    random_search_result = optimizer.random_search(n_iter=1000)
    print("Random Search Result:", random_search_result)

    # Perform simulated annealing optimization
    sa_result = optimizer.simulated_annealing()
    print("Simulated Annealing Result:", sa_result)

    # Perform genetic algorithm optimization
    ga_result = optimizer.genetic_algorithm(n_generations=100, population_size=50)
    print("Genetic Algorithm Result:", ga_result)

    # Perform particle swarm optimization
    pso_result = optimizer.particle_swarm_optimization(swarmsize=100, maxiter=200)
    print("Particle Swarm Optimization Result:", pso_result)

    # Save the optimization model
    model_dir = 'models/'
    optimizer.save_model(model_dir)
