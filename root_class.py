from numpy import where, clip, logical_and, ones, array, ceil
from numpy.random import uniform
from copy import deepcopy


class Root:
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0         # index of min problem; ID_BEST
    ID_MAX_PROB = -1        # index of max problem; ID_WORST

    EPSILON = 10E-10

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=(1,2) , verbose=False, minimize_objFunc=True):
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
        self.minimize_objFunc = minimize_objFunc

        try:
            self.lb = lb * ones(problem_size)
            self.ub = ub * ones(problem_size)
        except (TypeError, ValueError) as e:                                                                                # lb, ub must be integers and problem_size must be a tuple of integers
            print(str(e))
            exit(0)

    def create_agent(self):
        position = uniform(self.lb, self.ub)
        fitness = self.get_position_fitness(position=position)
        return [position, fitness]

    def get_position_fitness(self, position=None): return self.obj_func(position) if self.minimize_objFunc else 1.0 / (self.obj_func(position) + self.EPSILON)

    def get_sorted_pop_and_global_best_agent(self, pop=None, id_best=None):                                 # biggest fitness --> smallest fitness
        """
        Sort population and return the sorted population and the best position 
        Method is only called once (during initialization of the slime mould agents)
        """
        pop.sort(key=lambda slime: slime.fitness, reverse=True)
        return pop, deepcopy(pop[id_best])

    def amend_position(self, position=None):
        return clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return where(logical_and(self.lb <= position, position <= self.ub), position, uniform(self.lb, self.ub))

    def update_sorted_population_and_global_best_agent(self, pop=None, id_best=None, g_best=None):
        """ 
        Sort the population, update the global best slime agent, and retun them
        """
        pop.sort(key=lambda slime: slime.fitness, reverse=True)
        current_best = pop[id_best]
        g_best = deepcopy(current_best) if current_best.get_fitness() < g_best.get_fitness() else deepcopy(g_best)            # Compare global best slime mould w/ this batch's best slime mould
        return sorted_pop, g_best