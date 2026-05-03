class Optimizer:
    def step(self, layers: list):
        raise NotImplementedError

    def zero_grad(self, layers: list):
        for layer in layers:
            if hasattr(layer, 'gradients'):
                layer.gradients = np.zeros_like(layer.gradients)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, l1_lambda=None, l2_lambda=None):
        self.learning_rate = learning_rate
        self.l1_lambda     = l1_lambda
        self.l2_lambda     = l2_lambda

    def step(self, layers, batch_size=1):
        for layer in layers:
            if not hasattr(layer, 'weights'):
                continue
            grad = layer.gradients / batch_size
            if self.l1_lambda:
                l1 = self.l1_lambda * np.sign(layer.weights)
                l1[:, 0] = 0
                grad += l1
            if self.l2_lambda:
                l2 = 2 * self.l2_lambda * layer.weights
                l2[:, 0] = 0
                grad += l2
            layer.weights -= self.learning_rate * grad


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, l1_lambda=None, l2_lambda=None):
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.l1_lambda     = l1_lambda
        self.l2_lambda     = l2_lambda
        self.velocity      = {}

    def step(self, layers, batch_size=1):
        for layer in layers:
            if not hasattr(layer, 'weights'):
                continue
            lid = id(layer)
            if lid not in self.velocity:
                self.velocity[lid] = np.zeros_like(layer.weights)
            grad = layer.gradients / batch_size
            if self.l1_lambda:
                l1 = self.l1_lambda * np.sign(layer.weights)
                l1[:, 0] = 0
                grad += l1
            if self.l2_lambda:
                l2 = 2 * self.l2_lambda * layer.weights
                l2[:, 0] = 0
                grad += l2
            self.velocity[lid] = self.momentum * self.velocity[lid] + self.learning_rate * grad
            layer.weights -= self.velocity[lid]


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, rho=0.9, eps=1e-8, l1_lambda=None, l2_lambda=None):
        self.learning_rate = learning_rate
        self.rho           = rho
        self.eps           = eps
        self.l1_lambda     = l1_lambda
        self.l2_lambda     = l2_lambda
        self.cache         = {}

    def step(self, layers, batch_size=1):
        for layer in layers:
            if not hasattr(layer, 'weights'):
                continue
            lid = id(layer)
            if lid not in self.cache:
                self.cache[lid] = np.zeros_like(layer.weights)
            grad = layer.gradients / batch_size
            if self.l1_lambda:
                l1 = self.l1_lambda * np.sign(layer.weights)
                l1[:, 0] = 0
                grad += l1
            if self.l2_lambda:
                l2 = 2 * self.l2_lambda * layer.weights
                l2[:, 0] = 0
                grad += l2
            self.cache[lid] = self.rho * self.cache[lid] + (1 - self.rho) * grad ** 2
            layer.weights -= self.learning_rate * grad / (np.sqrt(self.cache[lid]) + self.eps)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, l1_lambda=None, l2_lambda=None):
        self.learning_rate = learning_rate
        self.beta1         = beta1
        self.beta2         = beta2
        self.eps           = eps
        self.l1_lambda     = l1_lambda
        self.l2_lambda     = l2_lambda
        self.m             = {}
        self.v             = {}
        self.t             = 0

    def step(self, layers, batch_size=1):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, 'weights'):
                continue
            lid = id(layer)
            if lid not in self.m:
                self.m[lid] = np.zeros_like(layer.weights)
                self.v[lid] = np.zeros_like(layer.weights)
            grad = layer.gradients / batch_size
            if self.l1_lambda:
                l1 = self.l1_lambda * np.sign(layer.weights)
                l1[:, 0] = 0
                grad += l1
            if self.l2_lambda:
                l2 = 2 * self.l2_lambda * layer.weights
                l2[:, 0] = 0
                grad += l2
            self.m[lid] = self.beta1 * self.m[lid] + (1 - self.beta1) * grad
            self.v[lid] = self.beta2 * self.v[lid] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[lid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[lid] / (1 - self.beta2 ** self.t)
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
