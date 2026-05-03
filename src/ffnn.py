# Program F03_FFNN

# IDENTITAS
# Data Kelompok  : K01 - G03 - Geprek ML
# Data Anggota 1 : 13523021 - Muhammad Raihan Nazhim Oktana
# Data Anggota 2 : 13523044 - Muhammad Luqman Hakim
# Data Anggota 3 : 13523057 - Faqih Muhammad Syuhada
# Institusi      : Institut Teknologi Bandung (ITB)
# Mata Kuliah    : Pembelajaran Mesin (IF3270-24) Tahun 2026
# Detail Tugas   : Tugas Besar 1
# Hari & Tanggal : Rabu, 18 Maret 2026
# Deskripsi      : Implementasi FFNN (Feed Forward Neural Network)
# PIC F03        : 13523021 - Muhammad Raihan Nazhim Oktana

# KAMUS
# Numpy , Matplotlib : Library
# Functions : Module
# FFNN : Class (Feed Forward Neural Network)

# ALGORITMA
import numpy as np
from .functions import *
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

class FFNN :
    # DESKRIPSI LOKAL
    # FFNN (Feed Forward Neural Network) Implementation.

    # KAMUS LOKAL
    # ...
    
    # ALGORITMA

    def __init__(self: "FFNN",
                 dimension: list[int],
                 activation_functions: list,
                 loss_function: any = MSE,
                 learning_rate = 0.1,
                 weight_initialization = 'random_uniform',
                 seed=42, lower=0, upper=1,
                 mean=0, std=1, l1_lambda = None, l2_lambda = None
                 ):
        """
        initializes a feed forward neural network.
        dimension             : numbers of unit in each layer, starting from the input layer
                                up to the output layer.

        activation_functions  : activation function to use for each layer

        loss_function         : loss function to use
        weight_initialization : can be one of "zero", "random_uniform",
                                "random_normal", "xavier", "he"
        """
        if seed is not None:
            np.random.seed(seed)

        self.weights = []
        self.dimension = dimension

        for i in range(len(dimension) - 1):
            fan_in = dimension[i]
            fan_out = dimension[i + 1]

            shape = (fan_out, fan_in + 1)

            if weight_initialization == "zero":
                W = np.zeros(shape)

            elif weight_initialization == "random_uniform":
                W = np.random.uniform(lower, upper, size=shape)

            elif weight_initialization == "random_normal":
                W = np.random.normal(mean, std, size=shape)

            elif weight_initialization == "xavier":
                limit = np.sqrt(6 / (fan_in + fan_out))
                W = np.random.uniform(-limit, limit, size=shape)

            elif weight_initialization == "he":
                std = np.sqrt(2 / fan_in)
                W = np.random.normal(0, std, size=shape)

            else:
                raise ValueError(f"Unknown weight_initialization: {weight_initialization}")

            self.weights.append(W)

        self.gradients = [np.zeros_like(w) for w in self.weights]
        self.net = []
        self.out = []
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.delta = []
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def count_net_per_layer(self, last_layer : int) :
        # DESKRIPSI LOKAL
        # Count net value for each unit (neuron) in current layer.

        # KAMUS LOKAL
        # ...

        # ALGORITMA
        last = self.out[last_layer]
        current_layer = last_layer + 1
        result = np.hstack(
            [np.ones( (len(last), 1) ),
            last]
        )
        result = result @ self.weights[current_layer-1].T
        if len(self.net) >= current_layer + 1:
            self.net[current_layer] = result
        else:
            assert(len(self.net) == current_layer)
            self.net.append(result)
        return self

    def activate_layer(self, layer : int) :
        # DESKRIPSI LOKAL
        # Activate current layer using given activation function.

        # KAMUS LOKAL
        # ...

        # ALGORITMA
        result = self.activation_functions[layer-1].f(self.net[layer])
        if len(self.out) >= layer + 1:
            self.out[layer] = result
        else:
            assert(len(self.out) == layer)
            self.out.append(result)
        return self
    
    def predict(self, input_layer : np.ndarray) :
        # DESKRIPSI LOKAL
        # Forward Propagation process in FFNN (Feed Forward Neural Network).

        # KAMUS LOKAL
        # ...

        # ALGORITMA
        self.out = []
        self.net = []
        self.out.append(input_layer)
        self.net.append(input_layer)
        for i in range(1, len(self.weights)+1):
            self.count_net_per_layer(i-1).activate_layer(i)
        return self

    def result(self):
        return self.out[-1]
    
    def compute_delta_output_layer(self, t):
        if ((self.loss_function == CCE or self.loss_function == BCE) 
                and self.activation_functions[-1] == Softmax):
            self.delta.insert(0, (self.out[-1] - t).T)
            logging.debug(f'{t=}')
            o = self.out[-1]
            logging.debug(f'{o=}')
            logging.debug(f'out{self.delta=}')
            return

        o = self.out[-1]
        logging.debug(f'{t=}')
        logging.debug(f'{o=}')
        dL_do = self.loss_function.dL_do(t, o)
        logging.debug(f'{dL_do=}')
        if self.activation_functions[-1] == Softmax:
            J = Softmax.df_dx(o=o)
            dL_dnet = J.T @ dL_do
        else:
            dL_dnet = self.activation_functions[-1].df_dx(o=o) * dL_do
        logging.debug(f'{dL_dnet=}')
        self.delta.insert(0, dL_dnet)

    def compute_delta_hidden_layer(self, layer):
        o = self.out[layer]
        delta_next = self.delta[0]
        w_next = self.weights[layer][:, 1:]
        logging.debug(f'comhid_{layer=}')
        logging.debug(f'{o=}')
        logging.debug(f'{self.weights=}')
        logging.debug(f'{w_next=}')
        logging.debug(f'{delta_next=}')


        dnet = w_next.T @ delta_next
        logging.debug(f'{dnet=}')
        do_dnet = self.activation_functions[layer-1].df_dx(o=o)  
        logging.debug(f'{do_dnet=}')
        delta = np.atleast_2d(do_dnet).T * dnet
        logging.debug(f'{delta=}')

        self.delta.insert(0, delta)

    def _plot_histogram(self, data_source, layeridx: list[int], title_prefix: str):
        for idx in layeridx:
            if 0 <= idx < len(data_source):
                data = data_source[idx].flatten()
                plt.figure(figsize=(16, 9))
                plt.hist(data, bins=30, color='blue', edgecolor='black')
                plt.title(f"{title_prefix} - Layer {idx}")
                plt.xlabel("Value")
                plt.ylabel("Freq")
                plt.grid(alpha=0.5)
                plt.show()

    def plot_w(self, layeridx: list[int]):
        self._plot_histogram(self.weights, layeridx, "Weight Dist")

    def plot_g(self, layeridx: list[int]):
        self._plot_histogram(self.gradients, layeridx, "Gradient Dist")

    def count_validation_loss(self, validation_X, validation_y):
        copy = FFNN(self.dimension, self.activation_functions, self.loss_function,
                    weight_initialization='zero')
        copy.weights = [w.copy() for w in self.weights]
        copy.predict(validation_X)
        result = copy.result()
        loss = np.sum(copy.loss_function.L(validation_y, result))
        self.validation_loss.append(loss)

    def compute_gradient(self, layer):
        delta = self.delta[layer - 1]
        a_prev = np.hstack(
            (
                np.ones( (len(self.out[layer-1]), 1) ),
                self.out[layer - 1]
            )
        )
        logging.debug(f'compute delta layer {layer}')
        logging.debug(f'compute {delta=}')
        logging.debug(f'compute {a_prev=}')
        logging.debug(f'compute outer {delta @ a_prev}')
        self.gradients[layer - 1] += delta @ a_prev

    def apply_gradient(self, layer, grad):
        if self.l1_lambda is None and self.l2_lambda is None:
            self.weights[layer - 1] -= self.learning_rate * grad
            return
        l1 = (self.l1_lambda or 0) * np.sign(self.weights[layer - 1])
        l2 = 2 * (self.l2_lambda or 0) * self.weights[layer - 1]
        l1[:, 0] = 0
        l2[:, 0] = 0
        self.weights[layer - 1] -= self.learning_rate * (grad + l1 + l2)

    def train(self, X, y, epochs, batch_size=1, verbose=False, output_file=None, validation_X=[], validation_y=[]):
        n_samples = X.shape[0]
        batch_size = n_samples if batch_size is None else batch_size
        from tqdm import tqdm

        pbar = tqdm(range(epochs), desc="Training")
        # pbar = range(epochs)
        self.training_loss = []
        self.validation_loss = []

        for epoch in pbar:
            loss = 0
            idx = np.random.permutation(n_samples)
            X, y = X[idx], y[idx]
            last_layer = len(self.weights)

            for start in range(0, n_samples, batch_size):
                X_batch = X[start:start + batch_size]
                y_batch = y[start:start + batch_size]

                for layer in range(1, last_layer + 1):
                    self.gradients[layer - 1] = np.zeros_like(self.weights[layer - 1])

                self.delta = []
                self.predict(X_batch)
                loss += np.sum(self.loss_function.L(y_batch, self.out[-1][0]))

                self.compute_delta_output_layer(y_batch)
                for layer in reversed(range(1, last_layer)):
                    self.compute_delta_hidden_layer(layer)
                for layer in range(1, last_layer + 1):
                    self.compute_gradient(layer)

                for layer in range(1, last_layer + 1):
                    self.apply_gradient(layer, self.gradients[layer - 1] / len(X_batch))

            self.training_loss.append(loss)
            pbar.set_postfix(loss=f"{loss:.6f}")
            if verbose:
                logging.info(f"{loss:.6f}")
                if len(validation_X) > 0:
                    self.count_validation_loss(validation_X, validation_y)

            if output_file is not None:
                output_file.write(f"{epoch+1},{loss:.6f}\n")
                output_file.flush()

if __name__ == '__main__':
    a = FFNN([3,3,3,3], [Tanh, Sigmoid, Softmax], CCE, 0.1, l1_lambda=0., l2_lambda=0.0)
    a.train(np.array([[1,1,1],[-1,-1,-1],[2,-2,2]]), np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]), 1, verbose=True,
            validation_X=[[1,1,1],[-1,-1,-1],[2,-2,2]], validation_y=[[1, 0, 0],[0, 1, 0],[0,0,1]], batch_size=3)
    print(a.weights)
    a.predict([[1,1,1],[-1,-1,-1],[2,-2,2]])
    print(a.result())
    print(CCE.L(np.array([[1, 0, 0],[0, 1, 0],[0,0,1]]), a.result()))
    for net in a.net:
        print(f"{net=}")
    for o in a.out:
        print(f"{o=}")
    # print(a.validation_loss)
    # print(a.training_loss)
