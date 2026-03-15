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
from unit import Unit

class FFNN :
    # DESKRIPSI LOKAL
    # Forward Propagation implementation in FFNN (Feed Forward Neural Network).

    # KAMUS LOKAL
    # count_net_per_layer , activate_layer , forward_propagation : Function
    
    # ALGORITMA

    def __init__(self: "FFNN",
                 dimension: list[int],
                 activation_functions: list):
        """
        initializes a feed forward neural network.
        dimension: numbers of unit in each layer, starting from the input layer
                    up to the output layer.
        activation_functions: activation function to use for each layer
        """
        self.weights = [np.random.rand(dimension[i+1], dimension[i]+1) for i in range(0, len(dimension)-1)]
        self.net = []
        self.out = []
        self.activation_functions = activation_functions

    def count_net_per_layer(self, last_layer : np.ndarray | int) :
        # DESKRIPSI LOKAL
        # Count net value for each unit (neuron) in current layer.

        # KAMUS LOKAL
        # last_layer : np.ndarray | int
        # current_layer : list[Unit]
        # last : np.ndarray
        # len : Function
        # i : int (index)

        # ALGORITMA
        if (isinstance(last_layer , np.ndarray)) :
            last = last_layer
            current_layer = 0
        else:
            last = self.out[last_layer]
            current_layer = last_layer + 1
        result = np.hstack(
            [np.ones( (len(last), 1) ),
            last]
        )
        result = result @ self.weights[current_layer].T
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
        result = self.activation_functions[layer].f(self.net[layer])
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
        self.count_net_per_layer(input_layer).activate_layer(0)
        
        for i in range(1, len(self.weights)):
            self.count_net_per_layer(i-1).activate_layer(i)
        return self

    def result(self):
        return self.out[-1]
