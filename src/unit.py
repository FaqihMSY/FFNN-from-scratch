# Program F01_Unit

# IDENTITAS
# Data Kelompok  : K01 - G03 - Geprek ML
# Data Anggota 1 : 13523021 - Muhammad Raihan Nazhim Oktana
# Data Anggota 2 : 13523044 - Muhammad Luqman Hakim
# Data Anggota 3 : 13523057 - Faqih Muhammad Syuhada
# Institusi      : Institut Teknologi Bandung (ITB)
# Mata Kuliah    : Pembelajaran Mesin (IF3270-24) Tahun 2026
# Detail Tugas   : Tugas Besar 1
# Hari & Tanggal : Minggu, 15 Maret 2026
# Deskripsi      : Implementasi Unit FFNN (Feed Forward Neural Network)
# PIC F01        : 13523044 - Muhammad Luqman Hakim

# KAMUS
# Unit : Class (Unit FFNN - Single Neuron)

# ALGORITMA
from random import random
import numpy as np

class Unit :
    def __init__(self : "Unit" , dimension : int = 1 , initial_weights : np.ndarray = None) :
        '''
        dimension : Unit dimension
        error : error function
        net : net value
        out : output value
        '''
        if (initial_weights is not None) :
            self.dimension = len(initial_weights) + 1
            self.weights = initial_weights
            if (dimension != self.dimension) :
                raise ValueError("Specified dimension is different from initial weights dimension")
        elif (dimension > 0) :
            self.dimension = dimension
            rng = np.random.default_rng()
            self.weights = rng.uniform(size = dimension + 1)
        self.net = 0.0
        self.out = 0.0

    def predict(self : "Unit" , X : np.ndarray) -> np.ndarray :
        result = np.vstack((np.ones(len(X)) , X))
        result = np.linalg.multidot([result , self.weights])
        return result