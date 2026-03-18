# Program F03_Forward_Propagation

# IDENTITAS
# Data Kelompok  : K01 - G03 - Geprek ML
# Data Anggota 1 : 13523021 - Muhammad Raihan Nazhim Oktana
# Data Anggota 2 : 13523044 - Muhammad Luqman Hakim
# Data Anggota 3 : 13523057 - Faqih Muhammad Syuhada
# Institusi      : Institut Teknologi Bandung (ITB)
# Mata Kuliah    : Pembelajaran Mesin (IF3270-24) Tahun 2026
# Detail Tugas   : Tugas Besar 1
# Hari & Tanggal : Minggu, 15 Maret 2026
# Deskripsi      : Implementasi Forward Propagation FFNN (Feed Forward Neural Network)
# PIC F03        : 13523021 - Muhammad Raihan Nazhim Oktana

# KAMUS
# np : Numpy (Library)
# Unit : Class (Unit FFNN - Single Neuron)
# ForwardPropagation : Class (Forward Propagation FFNN - Feed Forward Neural Network)

# ALGORITMA
import numpy as np
from .functions import *
import matplotlib.pyplot as plt

class FFNN :
    # DESKRIPSI LOKAL
    # Forward Propagation implementation in FFNN (Feed Forward Neural Network).

    # KAMUS LOKAL
    # count_net_per_layer , activate_layer , forward_propagation : Function
    
    # ALGORITMA

    def __init__(self: "FFNN",
                 dimension: list[int],
                 activation_functions: list,
                 loss_function: any = MSE,
                 learning_rate = 0.1,
                 weight_initialization = 'random_uniform',
                 seed=42, lower=0, upper=1,
                 mean=0, std=1,
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

        for i in range(len(dimension) - 1):
            fan_in = dimension[i]
            fan_out = dimension[i + 1]

            shape = (fan_out, fan_in + 1)

            if weight_initialization == "zero":
                W = np.zeros(shape)

            elif weight_initialization == "random_uniform":
                W = np.random.uniform(lower, upper, size=shape)

            elif weight_initialization == "random_normal":
                std = np.sqrt(var)
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

    def count_net_per_layer(self, last_layer : int) :
        # DESKRIPSI LOKAL
        # Count net value for each unit (neuron) in current layer.

        # KAMUS LOKAL
        # last_layer : np.ndarray | int
        # current_layer : list[Unit]
        # last : np.ndarray
        # len : Function
        # i : int (index)

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

    def activate_layer(self,
                       layer : int) :
        # DESKRIPSI LOKAL
        # Activate current layer using given activation function.

        # KAMUS LOKAL
        # layer : list[Unit]
        # activation_function : Function (Linear , ReLU , Sigmoid , Tanh , Softmax)
        # net , res : np.ndarray
        # len : Function
        # i : int (index)

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
        # input_layer : np.ndarray
        # hidden_layers : list[list[Unit]]
        # output_layer : list[Unit]
        # activation_function_hidden_layers : list[any] (Activation Function)
        # activation_function_output_layer : any (Activation Function)
        # i : int (index)

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
        o = self.out[-1][0]
        # print(f'{t=}')
        # print(f'{o=}')
        dL_do = self.loss_function.dL_do(t, o)
        # print(f'{dL_do=}')
        dL_dnet = self.activation_functions[-1].df_dx(o) * dL_do
        # print(f'{dL_dnet=}')
        self.delta.insert(0, dL_dnet)

    def compute_delta_hidden_layer(self, layer):
        o = self.out[layer][0]
        delta_next = self.delta[0]
        w_next = self.weights[layer][:, 1:]
        # print(f'comhid_{layer=}')
        # print(f'{o=}')
        # print(f'{self.weights=}')
        # print(f'{w_next=}')
        # print(f'{delta_next=}')


        dnet = w_next.T @ delta_next
        # print(f'{dnet=}')
        do_dnet = self.activation_functions[layer-1].df_dx(o)  
        # print(f'{do_dnet=}')
        delta = do_dnet[:, np.newaxis] * dnet
        # print(f'{delta=}')

        self.delta.insert(0, delta)

    def update_weight(self, layer):
        delta = self.delta[layer-1]
        # print(f'{layer=}')
        # print(f'{delta=}')
        # print(f'{self.out=}')
        a_prev = np.hstack([[1], self.out[layer-1][0]])
        # print(f'{a_prev=}')

        grad = np.outer(delta, a_prev)
        # print(f'{grad=}')
        # print(f'{self.weights[layer-1]=}')

        self.gradients[layer-1] = grad
        self.weights[layer-1] -= self.learning_rate * grad
        # print(f'{self.weights[layer-1]=}')

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

    def train(self, X, y, epochs, verbose=False, output_file=None):
        n_samples = X.shape[0]
        print_freq = max(1, epochs // 100) 
        from tqdm import tqdm

        pbar = tqdm(range(epochs), desc="Training")
        loss_hist = []

        for epoch in pbar:
            loss = 0
            for i in range(n_samples):

                x = X[i:i+1]
                t = y[i:i+1]

                self.delta = []

                self.predict(x)
                loss += self.loss_function.L(t, self.out[-1][0])

                self.compute_delta_output_layer(t)

                last_layer = len(self.weights)

                for layer in reversed(range(1, last_layer)):
                    self.compute_delta_hidden_layer(layer)
                # print(f'{self.delta=}')
                for layer in range(1, last_layer+1):
                    self.update_weight(layer)
            loss_hist.append(loss)
            if verbose:
                pbar.set_postfix(loss=f"{loss[0][0]:.6f}")

            if output_file is not None:
                output_file.write(f"{epoch+1},{loss[0][0]:.6f}\n")
                output_file.flush()

if __name__ == '__main__':
    a = FFNN([3,1,1,1], [Sigmoid, Sigmoid, Sigmoid, Sigmoid, ], BCE, 0.1)
    a.train(np.array([[1,2,3],[-4,-5,-6],[7,8,9]]), np.array([[1],[0],[1]]), 5432, verbose=True)
    print(a.weights)
    a.predict([[1,2,3],[-4,-5,-6],[7,8,9]])
    print(a.result())
    print(MSE.L(np.array([[1],[0],[1]]), a.result()))
    for net, o in zip(a.net, a.out):
        print(net)
        print(o)
