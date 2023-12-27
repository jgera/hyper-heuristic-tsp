import random
import itertools
import logging
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of cities
num_cities = 10  # Adjust as needed

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random city coordinates
city_coordinates = np.random.rand(num_cities, 2) * 10  # Scale to a reasonable range

# Compute Euclidean distance matrix
distance_matrix = cdist(city_coordinates, city_coordinates, 'euclidean')

# Objective function: Total distance of the tour
def objective_function(solution, distance_matrix):
    total_distance = 0
    for i in range(len(solution) - 1):
        city1, city2 = solution[i], solution[i + 1]
        total_distance += distance_matrix[city1, city2]
    # Add distance from the last city back to the starting city
    total_distance += distance_matrix[solution[-1], solution[0]]
    return -total_distance  # Negative because we want to maximize

# Low-level heuristics
def swap_two_cities(solution):
    index1, index2 = random.sample(range(len(solution)), 2)
    solution[index1], solution[index2] = solution[index2], solution[index1]
    return solution

def reverse_segment(solution):
    index1, index2 = sorted(random.sample(range(len(solution)), 2))
    solution[index1:index2 + 1] = reversed(solution[index1:index2 + 1])
    return solution

# High-level heuristics
def random_high_level_heuristic():
    return random.choice([swap_two_cities, reverse_segment])

# Hyper-heuristic algorithm with logging
def hyper_heuristic(num_iterations, num_cities, distance_matrix):
    # Initialize a random solution as a permutation of cities
    current_solution = list(range(num_cities))
    random.shuffle(current_solution)

    fitness_history = []

    logger.info("Initial solution: %s", current_solution)
    logger.info("Initial fitness: %f", -objective_function(current_solution, distance_matrix))

    # Main loop
    for iteration in range(num_iterations):
        # Apply a high-level heuristic to select a low-level heuristic
        selected_llh = random_high_level_heuristic()

        # Apply the selected low-level heuristic to the current solution
        modified_solution = selected_llh(current_solution[:])

        # Evaluate the quality of the modified solution
        current_fitness = objective_function(current_solution, distance_matrix)
        modified_fitness = objective_function(modified_solution, distance_matrix)

        # Update the current solution based on the fitness improvement
        if modified_fitness > current_fitness:
            current_solution = modified_solution

        # Log iteration details
        logger.info("Iteration %d - Fitness: %f", iteration + 1, -modified_fitness)
        fitness_history.append(-modified_fitness)

    # Return the final solution
    logger.info("Final solution: %s", current_solution)
    logger.info("Final fitness: %f", -objective_function(current_solution, distance_matrix))

    # Plot fitness history
    plt.plot(range(1, num_iterations + 1), fitness_history, label="Fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Fitness Progress")
    plt.legend()
    plt.show()

    return current_solution

# Example of using the hyper-heuristic algorithm
num_iterations = 1000
final_solution = hyper_heuristic(num_iterations, num_cities, distance_matrix)
