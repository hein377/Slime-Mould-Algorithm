class SlimeAgent():

    def __init__(self, position=None, fitness=None, weight=None):
        self.position = position
        self.fitness = fitness
        self.weight = weight

    def set_position(self, position):
        self.position = position

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_weight(self, weight):
        self.weight = weight

    def get_position(self):
        return self.position
    
    def get_fitness(self):
        return self.fitness
    
    def get_weight(self):
        return self.weight