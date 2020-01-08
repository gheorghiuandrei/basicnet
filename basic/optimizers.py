import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def optimize(self, layer):
        for key in layer.params:
            layer.params[key] -= self.lr * layer.grads[key]
