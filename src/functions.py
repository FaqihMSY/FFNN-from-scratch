# Program F02_Functions

# IDENTITAS
# Data Kelompok  : K01 - G03 - Geprek ML
# Data Anggota 1 : 13523021 - Muhammad Raihan Nazhim Oktana
# Data Anggota 2 : 13523044 - Muhammad Luqman Hakim
# Data Anggota 3 : 13523057 - Faqih Muhammad Syuhada
# Institusi      : Institut Teknologi Bandung (ITB)
# Mata Kuliah    : Pembelajaran Mesin (IF3270-24) Tahun 2026
# Detail Tugas   : Tugas Besar 1
# Hari & Tanggal : Minggu, 15 Maret 2026
# Deskripsi      : Implementasi Activation Functions FFNN (Feed Forward Neural Network)
# PIC F02        : 13523044 - Muhammad Luqman Hakim

# KAMUS
# Linear , ReLU , Sigmoid , Tanh , Softmax : Class (Activation Functions)

# ALGORITMA
import numpy as np

class Linear :
    def f(x : np.ndarray) -> np.ndarray :
        return x
    def df_dx(x : np.ndarray) -> np.ndarray :
        return 1

class ReLU :
    def f(x : np.ndarray) -> np.ndarray :
        return np.where(x > 0 , x , 0)
    def df_dx(x : np.ndarray) -> np.ndarray :
        return np.where(x > 0 , 1 , 0)

class Sigmoid:
    def f(x : np.ndarray) -> np.ndarray :
        return 1 / (1 + np.exp(-x))
    def df_dx(x : np.ndarray) -> np.ndarray :
        sigm = Sigmoid.f(x)
        return sigm * (1 - sigm)

class Tanh :
    def f(x : np.ndarray) -> np.ndarray :
        e_pos = np.exp(x)
        e_neg = np.exp(-x)
        return (e_pos - e_neg) / (e_pos + e_neg)
    def df_dx(x : np.ndarray) -> np.ndarray :
        t = Tanh.f(x)
        return 1 - t**2

class Softmax:
    def f(x : np.ndarray) -> np.ndarray :
        e = np.exp(x)
        sum = np.sum(e)
        return e / sum
    def df_dx(x : np.ndarray) -> np.ndarray :
        sm = Softmax.f(x)
        partial = sm * (np.eye(len(x)) - np.vstack(sm))
        return partial

class MSE:
    def L(t, o):
        return 0.5 * np.power((t-o), 2)

    def dL_do(t, o):
        return o - t

class BCE:
    def L(t, o):
        eps = 1e-12
        o = np.clip(o, eps, 1 - eps)
        return -(t * np.log(o) + (1 - t) * np.log(1 - o))

    def dL_do(t, o):
        eps = 1e-12
        o = np.clip(o, eps, 1 - eps)
        return (o - t) / (o * (1 - o))

class CCE: 
    def L(t, o):
        eps = 1e-12
        o = np.clip(o, eps, 1 - eps)
        return -np.sum(t * np.log(o), axis=-1)

    def dL_do(t, o):
        eps = 1e-12
        o = np.clip(o, eps, 1 - eps)
        return -t / o
