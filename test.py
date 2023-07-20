'''
#Imports
import numpy as np
import matplotlib.pyplot as plt

#Helper Functions
def calcR(angle, x=4, y=6):
  return np.sqrt((x*y)/(x*np.cos(angle)**2 + y*np.sin(angle)**2))
def stretch_mesh(r, theta):
  return [ r[i] * calcR(theta[i][0]) for i in range(len(r)) ]

#Polar Mesh Grid
r = np.linspace(0, 1, 100)
print(r)
theta = np.linspace(0, 2*np.pi, 100)
print(theta)
r, theta = np.meshgrid(r, theta)
# Stretch Function
r = stretch_mesh(r, theta)
# Transform to Cartesian
X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = X*X - Y*Y
# Plot
plt.figure(dpi=200)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
plt.show()
'''

'''
from grid_class import Grid
my_grid = Grid()
my_grid.display_grid()
'''

from slime_class import SlimeAgent
import numpy as np
s1 = SlimeAgent(np.array([[0,0]]), 0, 0)
s2 = SlimeAgent(np.array([[1,1]]), 1, 0)
s3 = SlimeAgent(np.array([[2,2]]), 2, 0)

ls = [s2, s1, s3]
for slime in ls: print(slime, slime.get_fitness())
print()
#ls.sort(key=lambda slime: slime.fitness, reverse=True)
#for slime in ls: print(slime, slime.get_fitness())

from copy import deepcopy
def sort_pop_and_get_global_best_agent(pop=None, id_best=None):                                          # biggest fitness --> smallest fitness
    """
    Sort population and return the best position 
    Method is only called once (during initialization of the slime mould agents)
    """
    pop.sort(key=lambda slime: slime.fitness, reverse=True)
    return deepcopy(pop[id_best])

g_best = sort_pop_and_get_global_best_agent(ls, 0)
for slime in ls: print(slime, slime.get_fitness())
print(g_best.position)