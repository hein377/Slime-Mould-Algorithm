import sys

import slime_moulds_class
from grid_class import Grid

import math
from math import pi
import numpy as np
import random

# Problem Set-Up Parameters Initialization
GRID_SIZE = tuple(int(dimension) for dimension in sys.argv[1].split(','))                # GRID_SIZE = ( dim1, dim2, ..., dimn) <tuple of ints.
NUM_FOOD = int(sys.argv[2])                                                              # NUM_FOOD <integer>
FOODAREA_GRIDAREA_RATIO = 1/5
VARIANCE_THRESHOLD = 0.03

# SMA Parameters Initialization
lb = GRID_SIZE[0]                                                                       # lowerbound of position value
ub = GRID_SIZE[1]                                                                       # upperbound of position value
problem_size = (1, 2)

VERBOSE = False
EPOCH = 20
POPSIZE = 40

def create_circulate_grid(position, slime_scan_radius):
    r_vals = np.linspace(0, slime_scan_radius, slime_scan_radius//10) 
    theta_vals = np.linspace(0, 2*pi, 100)

    return circle_x_vals, circle_y_vals 

def obj_function_distance(position, slime_scan_radius, grid):                           # currently scanning w/ a propagating square wave; change to circular later

    return 0

# SMA
my_grid = Grid(GRID_SIZE, NUM_FOOD, FOODAREA_GRIDAREA_RATIO, VARIANCE_THRESHOLD)
my_grid.display_grid()
input()
obj_function_distance(np.array([[200, 200]]), 100, my_grid)
input()

# Slime Mould Algorithm
SMA = slime_moulds_class.BaseSMA(obj_function_distance, lb, ub, problem_size, VERBOSE, EPOCH, POPSIZE)