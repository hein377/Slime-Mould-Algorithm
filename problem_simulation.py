import sys

import numpy as np
import random

# Global Variables
GRID_SIZE = tuple(int(dimension) for dimension in sys.argv[1].split(','))       # GRID_SIZE = ( dim1, dim2, ..., dimn) <tuple of ints.
NUM_FOOD = int(sys.argv[2])                                                     # NUM_FOOD <integer>

# ASSUMES 2D GRID
def spawn_food(grid_size, num_food):
    width, height = grid_size
    grid = np.zeros(grid_size)
    food_coords = list(zip(random.sample(range(width), num_food), random.sample(range(height), num_food)))

    for x,y in food_coords: grid[x][y] = 1


spawn_food(GRID_SIZE, NUM_FOOD)