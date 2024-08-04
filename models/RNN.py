import numpy as np



class Helpers:
    def __init__(self):
        pass

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, da_next, at):
        return da_next * (1 - np.power(at, 2))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def softmax(self, x):  # assuming x is a 1d array
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def dsoftmax(self, dy, at, grads, params):

        grads['dWya'] += np.dot(dy, at.T)
        grads['dby'] += dy
        da_next = np.dot(params['Wya'].T, dy) + grads['da_next']

        return da_next


class RNN(Helpers):
    def __init__(self):
        super().__init__()

    def step_forward(self, xt, a_pre, params):

        at = self.tanh(np.dot(params["Wax"], xt) + np.dot(params["Waa"], a_pre) + params["ba"])
        yt = np.dot(params["Way"], self.softmax(at)) + params["by"]

        cache = (at, a_pre, xt, params)

        return (at, yt, cache)

    def forward_propagation(self, x, a0, params):

        caches = []
        n_x, m, T_x = x.shape
        n_y, n_a = params["Wya"].shape

        a = np.zeros((n_a, m, T_x), dtype=float)
        y_pred = np.zeros((n_y, m, T_x), dtype=float)

        a_next = a0

        for t in range(T_x):

            a_next, yt_pred, cache = self.step_forward(
                xt=x[:, :, t],
                a_pre=a_next,
                params=params
            )

            a[:, :, t] = a_next
            y_pred[:, :, t] = yt_pred
            caches.append(cache)

        caches = (caches, x)

        return (a, y_pred, caches)

    def compute_cost(self, y, y_pred):

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        cost = -np.sum(y * np.log(y_pred)) / y.shape[1]

        return cost

    def step_backward(self, dy, grads, cache):

        (at, a_pre, xt, params) = cache

        da_next = self.dsoftmax(
            dy=dy,
            at=at,
            grads=grads,
            params=params
        )
        dtanh = self.dtanh(da_next=da_next, at=at)
        d_xt = np.dot(params["Wax"].T, dtanh)
        dWax = np.dot(dtanh, xt.T)
        da_prev = np.dot(params["Waa"].T, dtanh)
        dWaa = np.dot(dtanh, a_pre.T)
        dba = np.sum(dtanh, axis=1, keepdims=True)

        grads = {
            "dxt": d_xt,
            "da_prev": da_prev,
            "dWax": dWax,
            "dWaa": dWaa,
            "dba": dba
        }

        return grads



    def backward_propagation(self, X, Y, params, caches):

        grads = {}
        (caches, x) = caches
        (a1, a0, x1, params) = caches[0]

        grads["dWax"] = np.zeros_like(params["Wax"])
        grads["dWaa"] = np.zeros_like(params["Waa"])
        grads["dWya"] = np.zeros_like(params["Wya"])
        grads["dba"] = np.zeros_like(params["ba"])
        grads["dby"] = np.zeros_like(params["by"])
        grads["da_next"] = np.zeros_like(a1)

        for t in reversed(range(len(X))):

            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            gradients = self.step_backward(dy, grads, params, x[t], a[t], a[t - 1])



        return grads






