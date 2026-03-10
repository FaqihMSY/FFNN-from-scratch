from random import random
import numpy as np
class Unit:
    def __init__(self, dimension=1, initial_weights=None):
        '''
        dimension: Unit dimension
        error: error function
        '''
        if not initial_weights is None:
            self.dimension = len(initial_weights)+1
            self.weights = initial_weights
            if dimension != self.dimension:
                raise Error("Specified dimension is different from initial weights dimension")
        elif dimension > 0:
            self.dimension = dimension
            rng = np.random.default_rng()
            self.weights = rng.uniform(size=dimension+1)

    def predict(self, X):
        result = np.vstack((np.ones(len(X)), X))
        result = np.linalg.multidot([result, self.weights])
        return result
