import numpy as np
class Linear:
    def f(x):
        return x
    def df_dx(x):
        return 1

class ReLU:
    def f(x):
        return np.where(x > 0, x, 0)
    def df_dx(x):
        return np.where(x > 0, 1, 0)

class Sigmoid:
    def f(x):
        return 1 / (1 + np.exp(-x))
    def df_dx(x):
        sigm = Sigmoid.f(x)
        return sigm * (1 - sigm)

class Tanh:
    def f(x):
        e_pos = np.exp(x)
        e_neg = np.exp(-x)
        return (e_pos - e_neg) / (e_pos + e_neg)
    def df_dx(x):
        e_pos = np.exp(x)
        e_neg = np.exp(-x)
        return np.pow(2 / (e_pos - e_neg), 2)

class Softmax:
    def f(x):
        e = np.exp(x)
        sum = np.sum(e)
        return e / sum
    def df_dx(x):
        sm = Softmax.f(x)
        partial = sm * (np.eye(len(x)) - np.vstack(sm))
        return partial
