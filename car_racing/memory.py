import random
from collections import deque


class Memory:
    """ Helper class for storing and retrieving transitions. """

    def __init__(self, capacity, batch_size, init_capacity=0):
        self.batch_size = batch_size
        self.init_capacity = init_capacity
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def num_of_samples(self):
        """ Returns number of samples stored in memory. """
        return len(self.memory)

    def is_initialized(self):
        """ Checks whether the memory contains the minimum required number of samples. """
        return self.num_of_samples() >= self.init_capacity

    def store(self, *args):
        """ Saves a transition. """
        self.memory.append(args)

    def sample(self):
        """ Returns a list of samples. """
        return random.sample(self.memory, self.batch_size)

    def flush(self):
        """ In case we want to empty the memory. """
        self.memory.clear()
