import numpy as np


class BinaryCrossEntropy:
    @staticmethod
    def forward(x, y):
        return -np.sum(y * np.log(x) + (1 - y) * np.log(1 - x))

    @staticmethod
    def backward(x, y):
        return -y / x + (1 - y) / (1 - x)
