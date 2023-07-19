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

from grid import Grid
my_grid = Grid()
my_grid.display_grid()