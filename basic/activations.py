import numpy as np


class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(x, 0)

    @staticmethod
    def backward(x, out_grads):
        return out_grads * (x > 0)


class Tanh:
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x, out_grads):
        return out_grads * (1 - np.tanh(x) ** 2)


class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x, out_grads):
        return out_grads * Sigmoid.forward(x) * (1 - Sigmoid.forward(x))
