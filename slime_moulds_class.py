from numpy.random import uniform, choice
from numpy import abs, zeros, log10, where, arctanh, tanh

from root_class import Root
from slime_class import SlimeAgent

class BaseSMA(Root):
    """
        Modified version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
        Notes:
            + Selected 2 unique and random solution to create new solution (not to create variable) --> remove third loop in original version
            + Check bound and update fitness after each individual move instead of after the whole population move in the original version
            + We always maximize the fitness value and by default maximize the objective function (fitness value proportional to objective_function), unless specified by the user setting minimize_objFunc=True
    """
    
    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=(1,2), verbose=False, minimize_objFunc=False, epoch=75, pop_size=100, z=0.03):
        Root.__init__(self, obj_func, lb, ub, problem_size, verbose, minimize_objFunc)
        self.epoch = epoch
        self.pop_size = pop_size
        self.z = z

        self.solution = None
        self.pop = None

    def create_agent(self):
        pos = uniform(self.lb, self.ub)
        fit = self.get_position_fitness(pos, self.minimize_objFunc)
        weight = zeros(self.problem_size)
        return SlimeAgent(pos, fit, weight)
    
    def update_slime_weights(self):
        s = self.pop[0].get_fitness() - self.pop[-1].get_fitness() + self.EPSILON                                    # Plus epsilon to avoid denominator zero

        for i in range(0, self.pop_size):                                                                          # Eq.(2.5)
            if i <= int(self.pop_size / 2): self.pop[i].set_weight(1 + uniform(0, 1, self.problem_size) * log10((self.pop[0].get_fitness() - self.pop[i].get_fitness()) / s + 1))
            else: self.pop[i].set_weight(1 - uniform(0, 1, self.problem_size) * log10((self.pop[0].get_fitness() - self.pop[i].get_fitness()) / s + 1))

    def train(self):
        self.pop = [self.create_agent() for _ in range(self.pop_size)]
        g_best = self.sort_pop_and_get_global_best_agent(self.pop, self.ID_FIT, self.ID_MIN_PROB)      # Eq.(2.6)

        for epoch in range(self.epoch):
            self.update_slime_weights()                                                                            # Update the fitness weight of each slime mold

            a = arctanh(-((epoch + 1) / self.epoch) + 1)                                                           # Eq.(2.4)
            b = 1 - (epoch + 1) / self.epoch

            # Update the Position of search agents
            for i in range(0, self.pop_size):
                if uniform() < self.z:  # Eq.(2.7)
                    pos_new = uniform(self.lb, self.ub)
                else:
                    p = tanh(abs(self.pop[i].get_fitness() - g_best.get_fitness()))                                  # Eq.(2.2)
                    vb = uniform(-a, a, self.problem_size)                                                         # Eq.(2.3)
                    vc = uniform(-b, b, self.problem_size)

                    # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                    id_a, id_b = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)

                    pos_1 = g_best.get_position() + vb * (self.pop[i].get_weight() * self.pop[id_a].get_position() - self.pop[id_b].get_position())
                    pos_2 = vc * self.pop[i].get_position()
                    pos_new = where(uniform(0, 1, self.problem_size) < p, pos_1, pos_2)

                # Check bound and re-calculate fitness after each individual move
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_position_fitness(pos_new)
                self.pop[i].set_position(pos_new)
                self.pop[i].set_fitness(fit_new)

            # Sorted population and update the global best
            g_best = self.update_sorted_population_and_get_global_best_agent(self.pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best.get_fitness())
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best.get_fitness()))

        self.solution = g_best
        return g_best.get_position(), g_best.get_fitness(), self.loss_train