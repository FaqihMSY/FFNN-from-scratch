class LinearLayer:
    def __init__(self,
                 fan_in: int,
                 fan_out: int,
                 weight_initialization='xavier',
                 seed=42, lower=0, upper=1,
                 mean=0, std=1,
                 l1_lambda=None, l2_lambda=None):

        if seed is not None:
            np.random.seed(seed)

        self.fan_in    = fan_in
        self.fan_out   = fan_out
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        shape = (fan_out, fan_in + 1)

        if weight_initialization == 'zero':
            self.weights = np.zeros(shape)
        elif weight_initialization == 'random_uniform':
            self.weights = np.random.uniform(lower, upper, size=shape)
        elif weight_initialization == 'random_normal':
            self.weights = np.random.normal(mean, std, size=shape)
        elif weight_initialization == 'xavier':
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = np.random.uniform(-limit, limit, size=shape)
        elif weight_initialization == 'he':
            std = np.sqrt(2 / fan_in)
            self.weights = np.random.normal(0, std, size=shape)
        else:
            raise ValueError(f"Unknown weight_initialization: {weight_initialization}")

        self.gradients = np.zeros_like(self.weights)
        self.input     = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = np.hstack((np.ones((len(x), 1)), x))
        return self.input @ self.weights.T

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.gradients += grad.T @ self.input
        return grad @ self.weights[:, 1:]

    def step(self, learning_rate, batch_size=1):
        grad = self.gradients / batch_size
        if self.l1_lambda is None and self.l2_lambda is None:
            self.weights -= learning_rate * grad
        else:
            l1 = (self.l1_lambda or 0) * np.sign(self.weights)
            l2 = 2 * (self.l2_lambda or 0) * self.weights
            l1[:, 0] = 0
            l2[:, 0] = 0
            self.weights -= learning_rate * (grad + l1 + l2)
        self.gradients = np.zeros_like(self.weights)

    def save(self) -> dict:
        return {
            'fan_in':    self.fan_in,
            'fan_out':   self.fan_out,
            'weights':   self.weights,
            'l1_lambda': self.l1_lambda or 0,
            'l2_lambda': self.l2_lambda or 0,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LinearLayer':
        layer = cls(
            fan_in=int(d['fan_in']),
            fan_out=int(d['fan_out']),
            weight_initialization='zero',
            l1_lambda=float(d['l1_lambda']) or None,
            l2_lambda=float(d['l2_lambda']) or None,
        )
        layer.weights = d['weights']
        return layer
