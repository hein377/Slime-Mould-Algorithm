import sys

import numpy as np
import random
import math

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Global Variables
GRID_SIZE = tuple(int(dimension) for dimension in sys.argv[1].split(','))       # GRID_SIZE = ( dim1, dim2, ..., dimn) <tuple of ints.
NUM_FOOD = int(sys.argv[2])                                                     # NUM_FOOD <integer>
FOODAREA_GRIDAREA_RATIO = 1/3
VARIANCE_THRESHOLD = 0.25

# ASSUMES 2D GRID
def display_grid(grid):
    plt.imshow(grid)
    plt.show()

'''
Link: https://en.wikipedia.org/wiki/Gaussian_function
f(x,y) = A * e^(-( (x-x0)^2/(2*sigma_x^2) + (y-y0)^2/(2*sigma_y^2) ))
Assume A=1, sigma_x=sigma_y=sigma --> f(x,y) = e^(-1/(2*sigma^2) * ( (x-x0)^2 + (y-y0)^2 ))

Let's say we want variance to be our 0.25 threshold; in other words, at z = 0.25, we obtain the equation (x-x0)^2 + (y-y0)^2 = variance^2.
This means within the circle centered at (x0,y0) w/ a radius of variance, f(x,y) = z = probability of having food >= 0.25

Solving for sigma, then, yields sigma = ( variance^2 / (-2*ln(z)) )^0.5 where z = desired threshold at (x-x0)^2 + (y-y0)^2 = variance^2
Substituting this value of sigma will then satisfy our condition of having our threshold be at the given circle w/ a radius of variance.

Note: variance does not refer to the mathematical variance, it is simply used to determine how far left/right/up/down we must move from the
center of the food source in order to meet the foodArea_gridArea_ratio given the number of food sources there are.
'''

def calculate_variance(grid_area, food_grid_ratio, num_food): return round((math.sqrt(grid_area*food_grid_ratio // num_food)-1) // 2)

def calculate_sigma(variance, z_val): return math.sqrt(variance**2 / -2*math.log(z_val))

def probability_func(x_vals, y_vals, center_x, center_y, sigma): return np.exp(-1/(2*sigma**2) * ((x_vals-center_x)**2+(y_vals-center_y)**2))

def create_food(grid, center_x, center_y, variance, sigma):
    print(center_x)
    print(center_y)

    food_x_axis = np.arange(center_x - variance, center_x + variance+1, 1)
    food_y_axis = np.arange(center_y - variance, center_y + variance+1, 1) 
    x_vals, y_vals = np.meshgrid(food_x_axis, food_y_axis)

    probabilities = probability_func(x_vals, y_vals, center_x, center_y, sigma)
    print(x_vals)
    print(y_vals)
    print(probabilities)
    input()

    ax = plt.axes(projection="3d")
    ax.scatter(x_vals, y_vals, probabilities, marker='^')
    ax.scatter(x_vals, y_vals, 0.25)
    plt.show()

def create_grid(grid_size, num_food, food_grid_ratio, variance_threshold):
    width, height = grid_size
    grid = np.zeros(grid_size)
    food_coords = list(zip(random.sample(range(width), num_food), random.sample(range(height), num_food)))
    variance = calculate_variance(width*height, food_grid_ratio, num_food)
    sigma = calculate_sigma(variance, variance_threshold)

    for x,y in food_coords: 
        create_food(grid, x, y, variance, sigma)
        input()

    return grid

grid = create_grid(GRID_SIZE, NUM_FOOD, FOODAREA_GRIDAREA_RATIO, VARIANCE_THRESHOLD)
#displ