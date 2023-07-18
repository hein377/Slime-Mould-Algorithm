from numpy import where, clip, logical_and, ones, array, ceil
from numpy.random import uniform
from copy import deepcopy


class Root:
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0         # min problem
    ID_MAX_PROB = -1        # max problem

    ID_POS = 0              # Position
    ID_FIT = 1              # Fitness

    EPSILON = 10E-10

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=(1,2) , verbose=False):
        """
        Parameters
        ----------
        obj_func : function
        lb : int
        ub : int
        problem_size : tuple, optional
        batch_size: int, optional
        verbose : bool, optional
        """
        
        self.obj_func = obj_func
        self.problem_size = problem_size
        self.verbose = verbose

        try:
            self.lb = lb * ones(problem_size)
            self.ub = ub * ones(problem_size)
        except (TypeError, ValueError) as e:                        # lb, ub must be integers and problem_size must be a tuple of integers
            print(str(e))
            exit(0)

    def create_agent(self, minmax=0):
        """ Return the position position with 2 element: position of position and fitness of position
        Parameters
        ----------
        minmax
            0 - minimum problem, else - maximum problem
        """
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        return [position, fitness]

    def get_fitness_position(self, position=None, minmax=0):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :param minmax: 0- min problem, 1 - max problem
        :return:
        """
        return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + self.EPSILON)

    def get_sorted_pop_and_global_best_solution(self, pop=None, id_fit=None, id_best=None):
        """ Sort population and return the sorted population and the best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        return sorted_pop, deepcopy(sorted_pop[id_best])

    def amend_position(self, position=None):
        return clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return where(logical_and(self.lb <= position, position <= self.ub), position, uniform(self.lb, self.ub))

    def update_sorted_population_and_global_best_solution(self, pop=None, id_best=None, g_best=None):
        """ Sort the population and update the current best position. Return the sorted population and the new current best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)
        return sorted_pop, g_best