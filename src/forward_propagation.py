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

class ForwardPropagation :
    # DESKRIPSI LOKAL
    # Forward Propagation implementation in FFNN (Feed Forward Neural Network).

    # KAMUS LOKAL
    # count_net_per_layer , activate_layer , forward_propagation : Function
    
    # ALGORITMA
    def count_net_per_layer(self : "ForwardPropagation" , last_layer : np.ndarray | list[Unit] , current_layer : list[Unit]) -> list[Unit] :
        # DESKRIPSI LOKAL
        # Count net value for each unit (neuron) in current layer.

        # KAMUS LOKAL
        # last_layer : any (np.ndarray | list[Unit])
        # current_layer : list[Unit]
        # last : np.ndarray
        # len : Function
        # i : int (index)

        # ALGORITMA
        if (isinstance(last_layer[0] , Unit)) :
            last = np.array([unit.out for unit in last_layer])
        else :
            last = last_layer
        for (i) in range (len(current_layer)) :
            current_layer[i].net = current_layer[i].predict([last])[0]
        return current_layer

    def activate_layer(self : "ForwardPropagation" , layer : list[Unit] , activation_function : any) -> list[Unit] :
        # DESKRIPSI LOKAL
        # Activate current layer using given activation function.

        # KAMUS LOKAL
        # layer : list[Unit]
        # activation_function : Function (Linear , ReLU , Sigmoid , Tanh , Softmax)
        # net , res : np.ndarray
        # len : Function
        # i : int (index)

        # ALGORITMA
        net = np.array([unit.net for unit in layer])
        res = activation_function.f(net)
        for i in range (len(layer)) :
            layer[i].out = res[i]
        return layer
    
    def forward_propagation(self : "ForwardPropagation" , input_layer : np.ndarray , hidden_layers : list[list[Unit]] , output_layer : list[Unit] , activation_function_hidden_layers : list[any] , activation_function_output_layer : any) -> list[np.ndarray , list[list[Unit]] , list[Unit]] :
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
        hidden_layers[0] = self.activate_layer(self.count_net_per_layer(input_layer , hidden_layers[0]) , activation_function_hidden_layers[0])
        for i in range (1 , len(hidden_layers)) :
            hidden_layers[i] = self.activate_layer(self.count_net_per_layer(hidden_layers[i - 1] , hidden_layers[i]) , activation_function_hidden_layers[i])
        output_layer = self.activate_layer(self.count_net_per_layer(hidden_layers[len(hidden_layers) - 1] , output_layer) , activation_function_output_layer)
        return [input_layer , hidden_layers , output_layer]