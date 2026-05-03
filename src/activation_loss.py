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
class ActivationLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(ActivationLayer):
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad


class ReLU(ActivationLayer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Sigmoid(ActivationLayer):
    def forward(self, x):
        self.out = np.where(x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Tanh(ActivationLayer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)


class Softmax(ActivationLayer):
    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = e / e.sum(axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        batch = self.out.shape[0]
        dx = np.zeros_like(self.out)
        for i in range(batch):
            s = self.out[i].reshape(-1, 1)
            J = np.diagflat(s) - s @ s.T
            dx[i] = J @ grad[i]
        return dx


class LossFunction:
    def forward(self, t: np.ndarray, o: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class MSE(LossFunction):
    def forward(self, t, o):
        self.t = t
        self.o = o
        return 0.5 * np.mean(np.sum((t - o) ** 2, axis=1))

    def backward(self):
        return (self.o - self.t) / self.t.shape[0]


class BCE(LossFunction):
    def forward(self, t, o):
        self.t = t
        self.o = o
        return -np.mean(t * np.log(o + 1e-9) + (1 - t) * np.log(1 - o + 1e-9))

    def backward(self):
        return (-(self.t / (self.o + 1e-9)) + (1 - self.t) / (1 - self.o + 1e-9)) / self.t.shape[0]


class CCE(LossFunction):
    def forward(self, t, o):
        self.t = t
        self.o = o
        return -np.mean(np.sum(t * np.log(o + 1e-9), axis=1))

    def backward(self):
        return -self.t / (self.o + 1e-9) / self.t.shape[0]


class SoftmaxCCE(LossFunction):
    def forward(self, x, t):
        self.t = t
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = e / e.sum(axis=1, keepdims=True)
        return -np.mean(np.sum(t * np.log(self.out + 1e-9), axis=1))

    def backward(self):
        return (self.out - self.t) / self.t.shape[0]

class SigmoidBCELoss(LossFunction):
    def forward(self, x, t):
        self.t = t
        self.out = np.where(x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)))
        return -np.mean(t * np.log(self.out + 1e-9) + (1 - t) * np.log(1 - self.out + 1e-9))

    def backward(self):
        return (self.out - self.t) / self.t.shape[0]
