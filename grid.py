import numpy as np
import random
import math

import matplotlib.pyplot as plt

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

class Grid:
    """
    Input Paramters
    ----------
    grid_size: tuple of ints
    num_food: int
    food_grid_ratio: double/float
    variance_thershold: double/float

    Intrinsic/Calculated Parameters
    ----------
    grid: numpy_array of grid_size
    sigma: double/float 
    variance: int
    """

    def calculate_variance(self, grid_area): return round((math.sqrt(grid_area*self.food_grid_ratio // self.num_food)-1) // 2)

    def calculate_sigma(self): return math.sqrt(self.variance**2 / (-2*math.log(self.variance_threshold)))               # variance_threshold = z_val

    def probability_func(self, x_vals, y_vals, center_x, center_y): return np.exp(-1/(2*self.sigma**2) * ((x_vals-center_x)**2+(y_vals-center_y)**2))

    def create_food(self, grid_width, grid_height, center_x, center_y):
        food_x_axis = np.arange(np.clip(center_x - self.variance, 0, grid_width), np.clip(center_x + self.variance + 1, 0, grid_width), 1)
        food_y_axis = np.arange(np.clip(center_y - self.variance, 0, grid_height), np.clip(center_y + self.variance + 1, 0, grid_height), 1)
        x_vals, y_vals = np.meshgrid(food_x_axis, food_y_axis)

        #display_probability_func(x_vals, y_vals, center_x, center_y, sigma)

        probabilities = self.probability_func(x_vals, y_vals, center_x, center_y)

        for row in range(probabilities.shape[0]):
            for col in range(probabilities.shape[1]):
                xval, yval = x_vals[row][col], y_vals[row][col]
                if(random.uniform(0,1) <= probabilities[row][col]): self.grid[xval][yval] = 1

    def create_grid(self):
        width, height = self.grid_size
        self.grid = np.zeros(self.grid_size)
        food_coords_centers = list(zip(random.sample(range(width), self.num_food), random.sample(range(height), self.num_food)))
        self.variance = self.calculate_variance(width*height)
        self.sigma = self.calculate_sigma()

        for center_x, center_y in food_coords_centers: self.create_food(width, height, center_x, center_y)

    def __init__(self, grid_size=(500,500), num_food=10, foodArea_gridArea_ratio=1/5, variance_threshold=0.03):
        self.grid_size = grid_size
        self.num_food = num_food
        self.food_grid_ratio = foodArea_gridArea_ratio
        self.variance_threshold = variance_threshold
        self.create_grid()

    def display_probability_func(self, x_vals, y_vals, center_x, center_y):
        probabilities = self.probability_func(x_vals, y_vals, center_x, center_y)

        ax = plt.axes(projection="3d")
        ax.scatter(x_vals, y_vals, probabilities, marker='^')
        ax.scatter(x_vals, y_vals, 0.25)
        plt.show()

    # ASSUMES 2D GRID
    def display_grid(self):
        plt.imshow(self.grid)
        plt.show()