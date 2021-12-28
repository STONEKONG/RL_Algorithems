import numpy as np 
import random

class Experience_Buffer():

    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        idx = len(experience) + len(self.buffer) - self.buffer_size
        if idx >= 0:
            self.buffer[0:idx] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(
            np.array(
                random.sample(self.buffer, size)
            ),
            [size, 5]
        )