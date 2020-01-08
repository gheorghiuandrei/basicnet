import numpy as np


class Model:
    def __init__(self, network, loss, optimizer, batch_size):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size

    def train(self, X, y, epochs):
        y = y.reshape(-1, 1)

        for i in range(epochs):
            indices = np.random.permutation(X.shape[0])

            for j in range(0, X.shape[0], self.batch_size):
                batch_indices = indices[j : j + self.batch_size]
                outputs = self._forward(X[batch_indices])
                grads = self.loss.backward(outputs.pop(), y[batch_indices])

                for output, layer in zip(
                    reversed(outputs), reversed(self.network)
                ):
                    grads = layer.backward(output, grads)

                    if hasattr(layer, "params"):
                        self.optimizer.optimize(layer)

    def predict(self, X):
        return self._forward(X)[-1]

    def _forward(self, X):
        outputs = [X]

        for layer in self.network:
            outputs.append(layer.forward(outputs[-1]))

        return outputs
