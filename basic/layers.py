import numpy as np


class FC:
    def __init__(self, in_features, out_features):
        bound = 1 / in_features ** 0.5
        weight_size = (out_features, in_features)
        bias_size = out_features
        self.params = {
            "weight": np.random.uniform(-bound, bound, weight_size),
            "bias": np.random.uniform(-bound, bound, bias_size),
        }
        self.grads = {"weight": None, "bias": None}

    def forward(self, x):
        return x @ self.params["weight"].T + self.params["bias"]

    def backward(self, x, out_grads):
        in_grads = out_grads @ self.params["weight"]
        self.grads["weight"] = out_grads.T @ x
        self.grads["bias"] = out_grads.sum(0)

        return in_grads
