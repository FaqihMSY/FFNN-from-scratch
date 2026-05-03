# Program F02_Functions

# IDENTITAS
# Data Kelompok  : K01 - G03 - Geprek ML
# Data Anggota 1 : 13523021 - Muhammad Raihan Nazhim Oktana
# Data Anggota 2 : 13523044 - Muhammad Luqman Hakim
# Data Anggota 3 : 13523057 - Faqih Muhammad Syuhada
# Institusi      : Institut Teknologi Bandung (ITB)
# Mata Kuliah    : Pembelajaran Mesin (IF3270-24) Tahun 2026
# Detail Tugas   : Tugas Besar 1
# Hari & Tanggal : Rabu, 18 Maret 2026
# Deskripsi      : Implementasi Activation Functions FFNN (Feed Forward Neural Network)
# PIC F02        : 13523044 - Muhammad Luqman Hakim

# KAMUS
# Numpy : Library
# Linear , ReLU , Sigmoid , Tanh , Softmax : Class (Activation Functions)
# MSE , BCE , CCE : Class (Loss Functions)

# ALGORITMA
import numpy as np
class Linear :
    def f(x : np.ndarray) -> np.ndarray :
        return x
    def df_dx(x : np.ndarray = None, o : np.ndarray = None) -> np.ndarray :
        if x is None and o is None:
            raise Exception("No value provided to df_dx")
        elif o is None:
            return np.ones_like(x)
        else:
            return np.ones_like(o)

class ReLU :
    def f(x : np.ndarray) -> np.ndarray :
        return np.where(x > 0 , x , 0)
    def df_dx(x : np.ndarray = None, o : np.ndarray = None) -> np.ndarray :
        if x is None and o is None:
            raise Exception("No value provided to df_dx")
        elif o is None:
            return np.where(x > 0 , 1 , 0)
        else:
            return np.where(o > 0 , 1 , 0)

class Sigmoid:
    def _positive_sigmoid(x : np.ndarray) -> np.ndarray :
        return 1 / (1 + np.exp(-x))


    def _negative_sigmoid(x : np.ndarray) -> np.ndarray :
        exp = np.exp(x)
        return exp / (exp + 1)

    def f(x : np.ndarray) -> np.ndarray :
        # source:
        # https://stackoverflow.com/questions/51976461/
        positive = x >= 0
        negative = ~positive

        result = np.empty_like(x, dtype=np.float32)
        result[positive] = Sigmoid._positive_sigmoid(x[positive])
        result[negative] = Sigmoid._negative_sigmoid(x[negative])
        return result

    def df_dx(x : np.ndarray = None, o : np.ndarray = None) -> np.ndarray :
        if x is None and o is None:
            raise Exception("No value provided to df_dx")
        elif o is None:
            sigm = Sigmoid.f(x)
            return sigm * (1 - sigm)
        else:
            return o * (1 - o)

class Tanh :
    def f(x : np.ndarray) -> np.ndarray :
        return np.tanh(x)
    def df_dx(x : np.ndarray = None, o : np.ndarray = None) -> np.ndarray :
        if x is None and o is None:
            raise Exception("No value provided to df_dx")
        elif o is None:
            t = np.tanh(x)
            return 1 - np.power(t, 2)
        else:
            return 1 - np.power(o, 2)

class Softmax:
    def f(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / np.sum(e, 1).reshape(-1, 1)

    def df_dx(x: np.ndarray = None, o: np.ndarray = None) -> np.ndarray:
        if x is None and o is None:
            raise Exception("No value provided to df_dx")
        elif o is None:
            sm = Softmax.f(x)
            return sm * (np.eye(len(sm)) - np.vstack(sm))
        else:
            return o * (np.eye(len(o)) - np.vstack(o))

class MSE:
    def L(t : np.ndarray , o : np.ndarray) -> np.ndarray :
        return 0.5 * np.power((t-o), 2)

    def dL_do(t : np.ndarray, o : np.ndarray) -> np.ndarray :
        return o - t

class BCE :
    def L(t : np.ndarray , o : np.ndarray) -> np.ndarray :
        eps = 1e-6
        o = np.clip(o , eps , 1 - eps)

        result = np.where(
            t == 1 , 
            -np.log(o) , 
            -np.log(1 - o)
        )
        return result

    def dL_do(t : np.ndarray , o : np.ndarray) -> np.ndarray :
        eps = 1e-6
        o = np.clip(o , eps , 1 - eps)
        return (o - t) / (o * (1 - o))

class CCE : 
    def L(t : np.ndarray , o : np.ndarray) -> np.ndarray :
        eps = 1e-9
        o = np.clip(o , eps , 1 - eps)
        return -np.sum(t * np.log(o) , axis = -1)

    def dL_do(t : np.ndarray , o : np.ndarray) -> np.ndarray :
        eps = 1e-9
        o = np.clip(o , eps , 1 - eps)
        return -t / o

ACTIVATION_MAP = {
    'Linear':  Linear,
    'ReLU':    ReLU,
    'Sigmoid': Sigmoid,
    'Tanh':    Tanh,
    'Softmax': Softmax,
}

LOSS_MAP = {
    'MSE': MSE,
    'CCE': CCE,
    'BCE': BCE,
}

