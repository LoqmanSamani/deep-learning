import numpy as np
import matplotlib.pyplot as plt
import copy
import h5py
import time
import scipy.io

""" Deep Feed Forward Neural Network (FFNN) with Dropout Regularization Technique for Boolean Classification Tasks"""


class DeepFeedForward(object):
    def __init__(self, hidden_activation="relu", output_activation="sigmoid", print_cost=False, visualize_cost=False, epsilon=1000, sigma=1e-15, seed=None):

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.print_cost = print_cost
        self.visualize_cost = visualize_cost
        self.epsilon = epsilon
        self.sigma = sigma
        self.seed = seed

        self.train_cost = []
        self.validation_cost = []
        self.test_cost = []
        self.time_ = []
        self.costs = []
        self.parameters_ = {}  # a dictionary for storing intermediate parameters
        self.parameters = {}  # a dictionary for storing final parameters
        self.train_predict = None
        self.validation_predict = None
        self.test_predict = None
        self.train_accuracy = None
        self.validation_accuracy = None
        self.test_accuracy = None

    def layer_sizes(self, X, Y, num_hidden_units):

        num_input_units = [X.shape[0]]
        num_output_units = [Y.shape[0]]
        num_hidden_units = list(num_hidden_units)

        num_layer_units = list(num_input_units + num_hidden_units + num_output_units)

        return num_layer_units

    def initialize_parameters(self, num_layer_units, seed):

        if seed:
            np.random.seed(seed)

        parameters = {}

        for l in range(1, len(num_layer_units)):
            parameters[f"W{l}"] = np.random.randn(num_layer_units[l], num_layer_units[l - 1]) * np.sqrt(2 / num_layer_units[l - 1])
            parameters[f"b{l}"] = np.zeros((num_layer_units[l], 1))

        return parameters

    def sigmoid(self, Z):

        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    def relu(self, Z):

        A = np.maximum(0, Z)
        cache = Z

        return A, cache

    def derivative_sigmoid(self, dA, cache):

        Z = cache
        A = 1 / (1 + np.exp(-Z))
        dZ = dA * A * (1 - A)

        assert (dZ.shape == Z.shape)

        return dZ

    def derivative_relu(self, dA, cache):

        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    def forward_propagation(self, X, parameters, hidden_activation, output_activation, keep_prob):

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the network

        for l in range(1, L):

            A_prev = A
            W = parameters[f"W{l}"]
            b = parameters[f"b{l}"]

            if hidden_activation == "relu" and keep_prob != 1:

                Z = np.dot(W, A_prev) + b
                linear_cache = (A_prev, W, b)
                A, activation_cache = self.relu(Z=Z)
                D = (np.random.rand(*A.shape) < keep_prob).astype(int)
                A = A * D / keep_prob

                caches.append((linear_cache, activation_cache, D))

            elif hidden_activation == "relu" and keep_prob == 1:

                Z = np.dot(W, A_prev) + b
                linear_cache = (A_prev, W, b)
                A, activation_cache = self.relu(Z=Z)

                caches.append((linear_cache, activation_cache))

            else:
                raise ValueError("hidden_activation must be 'relu'!")

        A_prev = A
        W = parameters[f"W{L}"]
        b = parameters[f"b{L}"]

        if output_activation == "sigmoid":

            Z = np.dot(W, A_prev) + b
            linear_cache = (A_prev, W, b)
            A, activation_cache = self.sigmoid(Z=Z)

            caches.append((linear_cache, activation_cache))
        else:
            raise ValueError("output_activation must be 'sigmoid'!")

        return A, caches

    def compute_cost(self, A, Y, sigma):

        m = Y.shape[1]

        cost = - np.sum(np.multiply(Y, np.log(A + sigma)) + np.multiply(1 - Y, np.log(1 - A + sigma))) / m
        cost = np.squeeze(cost)

        return cost

    def backward_propagation(self, A, Y, caches, hidden_activation, output_activation, keep_prob, sigma, dataset):

        grads = {}
        L = len(caches)  # number of network layers
        Y = Y.reshape(A.shape)

        dA = - (np.divide(Y, A + sigma) - np.divide(1 - Y, 1 - A + sigma))

        if output_activation == "sigmoid":

            current_cache = caches[-1]
            (linear_cache, activation_cache) = current_cache
            (A_prev, W, b) = linear_cache

            dZ = self.derivative_sigmoid(dA=dA, cache=activation_cache)

            m = A_prev.shape[1]
            if dataset == "train" and keep_prob != 1:

                dW = 1. / m * np.dot(dZ, A_prev.T)
                db = np.sum(dZ, axis=1, keepdims=True) / m
                dA_prev = np.dot(W.T, dZ)

            else:
                dW = 1. / m * np.dot(dZ, A_prev.T)
                db = np.sum(dZ, axis=1, keepdims=True) / m
                dA_prev = np.dot(W.T, dZ)

            grads[f"dW{L}"] = dW
            grads[f"db{L}"] = db
            grads[f"dA{L-1}"] = dA_prev

        else:
            raise ValueError("output_activation must be 'sigmoid'!")

        for l in reversed(range(L - 1)):

            if hidden_activation == "relu" and keep_prob != 1:

                current_cache = caches[l]
                (linear_cache, activation_cache, D) = current_cache
                (A_prev, W, b) = linear_cache
                dA = grads[f"dA{l+1}"] * D
                dA = dA / keep_prob

                dZ = self.derivative_relu(dA=dA, cache=activation_cache)

                m = A_prev.shape[1]
                dW = 1. / m * np.dot(dZ, A_prev.T)
                db = np.sum(dZ, axis=1, keepdims=True) / m
                dA_prev = np.dot(W.T, dZ)

            elif hidden_activation == "relu" and keep_prob == 1:

                current_cache = caches[l]
                (linear_cache, activation_cache) = current_cache
                (A_prev, W, b) = linear_cache

                dZ = self.derivative_relu(dA=grads[f"dA{l+1}"], cache=activation_cache)

                m = A_prev.shape[1]
                dW = 1. / m * np.dot(dZ, A_prev.T)
                db = np.sum(dZ, axis=1, keepdims=True) / m
                dA_prev = np.dot(W.T, dZ)

            else:
                raise ValueError("hidden_activation must be 'relu'!")

            grads[f"dW{l+1}"] = dW
            grads[f"db{l+1}"] = db
            grads[f"dA{l}"] = dA_prev

        return grads

    def update_parameters(self, parameters, grads, learning_rate):

        parameters_ = copy.deepcopy(parameters)
        L = len(parameters_) // 2

        for l in range(L):
            parameters_[f"W{l + 1}"] -= learning_rate * grads[f"dW{l + 1}"]
            parameters_[f"b{l + 1}"] -= learning_rate * grads[f"db{l + 1}"]

        return parameters_

    def predict(self, X, parameters, hidden_activation, output_activation, keep_prob):

        predicted = np.zeros((1, X.shape[1]))

        probabilities, caches = self.forward_propagation(

            X=X,
            parameters=parameters,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            keep_prob=keep_prob
        )

        predicted[0, :] = (probabilities > 0.5)

        return predicted

    def accuracy(self, predicted, Y):

        accuracy = np.sum((predicted == Y) / Y.shape[1])

        return accuracy

    def train(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, parameters, keep_prob,
              hidden_activation, output_activation, num_epochs, learning_rate, print_cost,  epsilon, sigma):

        print()
        for i in range(1, num_epochs + 1):
            tic = time.time()

            A, caches = self.forward_propagation(
                X=X_train,
                parameters=parameters,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                keep_prob=keep_prob
            )
            cost = self.compute_cost(
                A=A,
                Y=Y_train,
                sigma=sigma
            )
            self.train_cost.append(cost)

            if X_validation is not None:

                A_valid, _ = self.forward_propagation(
                    X=X_validation,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation,
                    keep_prob=keep_prob
                )

                cost_v = self.compute_cost(
                    A=A_valid,
                    Y=Y_validation,
                    sigma=sigma
                )
                self.validation_cost.append(cost_v)

            if X_test is not None:

                A_test, _ = self.forward_propagation(
                    X=X_test,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation,
                    keep_prob=keep_prob
                )
                cost_t = self.compute_cost(
                    A=A_test,
                    Y=Y_test,
                    sigma=sigma
                )
                self.test_cost.append(cost_t)

            grads = self.backward_propagation(
                A=A,
                Y=Y_train,
                caches=caches,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                keep_prob=keep_prob,
                sigma=sigma,
                dataset="train"
            )

            parameters = self.update_parameters(
                parameters=parameters,
                grads=grads,
                learning_rate=learning_rate
            )
            self.parameters_[i] = parameters  # store the intermediate parameters
            toc = time.time()
            self.time_.append(toc - tic)
            if print_cost and i % epsilon == 0:
                print(f"Epoch {i}; Cost: {round(cost, 5)}; Duration: {round(sum(self.time_[i - epsilon: i]), 5)} seconds")
            elif print_cost and i == 1:
                print(f"Epoch {i}; Cost: {round(cost, 5)}; Duration: {round(self.time_[0], 5)} seconds")
            if i == num_epochs and i % epsilon != 0:
                print(f"Epoch {i}; Cost: {round(cost, 5)}; Duration: {round(toc - tic, 5)} seconds")

        return parameters

    def model(self, X_train, Y_train, num_hidden_units, X_validation=None, Y_validation=None, X_test=None, Y_test=None, keep_prob=1.0, num_epochs=10000, learning_rate=1e-4,):

        num_layer_units = self.layer_sizes(
            X=X_train,
            Y=Y_train,
            num_hidden_units=num_hidden_units
        )
        parameters = self.initialize_parameters(
            num_layer_units=num_layer_units,
            seed=self.seed
        )

        if self.print_cost:

            print("___" * 40)
            print()
            print(f"NN_Model with {len(num_hidden_units) + 1} layers ({num_layer_units}); Number of Epochs: {num_epochs}; Learning Rate: {learning_rate}; Keep Probability: {keep_prob}")
            print()
            print("___" * 40)

            parameters_ = self.train(
                X_train=X_train,
                Y_train=Y_train,
                X_validation=X_validation,
                Y_validation=Y_validation,
                X_test=X_test,
                Y_test=Y_test,
                parameters=parameters,
                keep_prob=keep_prob,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                print_cost=self.print_cost,
                epsilon=self.epsilon,
                sigma=self.sigma
            )

            print()

            self.parameters = parameters_

            train_predicted = self.predict(
                X=X_train,
                parameters=self.parameters,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                keep_prob=keep_prob
            )
            train_accuracy = self.accuracy(predicted=train_predicted, Y=Y_train)
            self.train_predict = train_predicted
            self.train_accuracy = train_accuracy

            if X_validation is not None:
                valid_predicted = self.predict(
                    X=X_validation,
                    parameters=self.parameters,
                    hidden_activation=self.hidden_activation,
                    output_activation=self.output_activation,
                    keep_prob=keep_prob
                )
                valid_accuracy = self.accuracy(predicted=valid_predicted, Y=Y_validation)
                self.validation_predict = valid_predicted
                self.validation_accuracy = valid_accuracy

            if X_test is not None:
                test_predicted = self.predict(
                    X=X_test,
                    parameters=self.parameters,
                    hidden_activation=self.hidden_activation,
                    output_activation=self.output_activation,
                    keep_prob=keep_prob
                )
                test_accuracy = self.accuracy(predicted=test_predicted, Y=Y_test)
                self.test_predict = test_predicted
                self.test_accuracy = test_accuracy

            print("__" * 20)
            print()
            print(f"Train Accuracy: {self.train_accuracy}")
            if X_validation is not None:
                print(f"Validation Accuracy: {self.validation_accuracy}")
            if X_test is not None:
                print(f"Test Accuracy: {self.test_accuracy}")
            print()
            print("__" * 20)

            if self.visualize_cost:
                plt.plot(self.train_cost, label="Train Cost")
                if X_validation is not None:
                    plt.plot(self.validation_cost, label="Validation Cost")
                if X_test is not None:
                    plt.plot(self.test_cost, label="Test Cost")
                plt.title("Epoch vs. Cost")
                plt.xlabel(f"Epoch")
                plt.ylabel("Cost")
                plt.legend()
                plt.show()




def load_2D_dataset():
    data = scipy.io.loadmat("/home/samani/Documents/projects/deep-learning/data/data.mat")
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    return train_X, train_Y, test_X, test_Y

X_train, Y_train, X_test, Y_test = load_2D_dataset()


obj = DeepFeedForward(
    print_cost=True,
    visualize_cost=True
)


obj.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[10, 6, 4],
    X_test=X_test,
    Y_test=Y_test,
    keep_prob=1.0,
    num_epochs=30000,
    learning_rate=0.2
)


"""
_______________________________________________________________________________________________________________

NN_Model with 4 layers ([2, 10, 6, 4, 1]); Number of Epochs: 30000; Learning Rate: 0.2; Keep Probability: 1.0

_______________________________________________________________________________________________________________

Epoch 1; Cost: 0.69659; Duration: 0.00154 seconds
Epoch 1000; Cost: 0.1967; Duration: 0.55986 seconds
Epoch 2000; Cost: 0.18858; Duration: 0.51797 seconds
Epoch 3000; Cost: 0.18428; Duration: 0.65272 seconds
Epoch 4000; Cost: 0.17708; Duration: 0.66839 seconds
Epoch 5000; Cost: 0.16746; Duration: 0.7067 seconds
Epoch 6000; Cost: 0.1643; Duration: 0.68494 seconds
Epoch 7000; Cost: 0.16741; Duration: 0.64018 seconds
Epoch 8000; Cost: 0.14486; Duration: 0.62946 seconds
Epoch 9000; Cost: 0.12436; Duration: 0.68897 seconds
Epoch 10000; Cost: 0.12971; Duration: 0.66281 seconds
Epoch 11000; Cost: 0.12989; Duration: 0.66275 seconds
Epoch 12000; Cost: 0.12248; Duration: 0.70606 seconds
Epoch 13000; Cost: 0.12924; Duration: 0.70172 seconds
Epoch 14000; Cost: 0.10991; Duration: 0.71361 seconds
Epoch 15000; Cost: 0.10952; Duration: 0.79084 seconds
Epoch 16000; Cost: 0.11909; Duration: 0.69514 seconds
Epoch 17000; Cost: 0.10503; Duration: 0.71324 seconds
Epoch 18000; Cost: 0.10125; Duration: 0.64953 seconds
Epoch 19000; Cost: 0.0913; Duration: 0.66651 seconds
Epoch 20000; Cost: 0.07684; Duration: 0.70558 seconds
Epoch 21000; Cost: 0.06996; Duration: 0.69604 seconds
Epoch 22000; Cost: 0.06292; Duration: 0.7127 seconds
Epoch 23000; Cost: 0.05743; Duration: 0.69257 seconds
Epoch 24000; Cost: 0.0617; Duration: 0.70362 seconds
Epoch 25000; Cost: 0.05172; Duration: 0.66037 seconds
Epoch 26000; Cost: 0.05098; Duration: 0.66948 seconds
Epoch 27000; Cost: 0.04972; Duration: 0.70505 seconds
Epoch 28000; Cost: 0.04235; Duration: 0.71565 seconds
Epoch 29000; Cost: 0.03974; Duration: 0.71654 seconds
Epoch 30000; Cost: 0.04259; Duration: 0.68953 seconds

________________________________________

Train Accuracy: 0.9905213270142181
Test Accuracy: 0.9299999999999999

________________________________________
"""

obj.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[10, 6, 4],
    X_test=X_test,
    Y_test=Y_test,
    keep_prob=0.90,
    num_epochs=30000,
    learning_rate=0.03
)

"""
________________________________________________________________________________________________________________________

NN_Model with 4 layers ([2, 10, 6, 4, 1]); Number of Epochs: 30000; Learning Rate: 0.03; Keep Probability: 0.9

________________________________________________________________________________________________________________________

Epoch 1; Cost: 0.63225; Duration: 0.00231 seconds
Epoch 1000; Cost: 0.28973; Duration: 0.79515 seconds
Epoch 2000; Cost: 0.26113; Duration: 0.75354 seconds
Epoch 3000; Cost: 0.26852; Duration: 0.74595 seconds
Epoch 4000; Cost: 0.29629; Duration: 0.76906 seconds
Epoch 5000; Cost: 0.23164; Duration: 0.74274 seconds
Epoch 6000; Cost: 0.25911; Duration: 0.73669 seconds
Epoch 7000; Cost: 0.25446; Duration: 0.73892 seconds
Epoch 8000; Cost: 0.25524; Duration: 0.74172 seconds
Epoch 9000; Cost: 0.21809; Duration: 0.73845 seconds
Epoch 10000; Cost: 0.26207; Duration: 0.74263 seconds
Epoch 11000; Cost: 0.24422; Duration: 0.73928 seconds
Epoch 12000; Cost: 0.24087; Duration: 0.73462 seconds
Epoch 13000; Cost: 0.20265; Duration: 0.73462 seconds
Epoch 14000; Cost: 0.22455; Duration: 0.73974 seconds
Epoch 15000; Cost: 0.21648; Duration: 0.75855 seconds
Epoch 16000; Cost: 0.24803; Duration: 0.74206 seconds
Epoch 17000; Cost: 0.22776; Duration: 0.7415 seconds
Epoch 18000; Cost: 0.20949; Duration: 0.73692 seconds
Epoch 19000; Cost: 0.21668; Duration: 0.73412 seconds
Epoch 20000; Cost: 0.24593; Duration: 0.73596 seconds
Epoch 21000; Cost: 0.22044; Duration: 0.73376 seconds
Epoch 22000; Cost: 0.21175; Duration: 0.74701 seconds
Epoch 23000; Cost: 0.19101; Duration: 0.73679 seconds
Epoch 24000; Cost: 0.24777; Duration: 0.74162 seconds
Epoch 25000; Cost: 0.20726; Duration: 0.74357 seconds
Epoch 26000; Cost: 0.21971; Duration: 0.73904 seconds
Epoch 27000; Cost: 0.18586; Duration: 0.73934 seconds
Epoch 28000; Cost: 0.21193; Duration: 0.75058 seconds
Epoch 29000; Cost: 0.23032; Duration: 0.76701 seconds
Epoch 30000; Cost: 0.22846; Duration: 0.74741 seconds

________________________________________

Train Accuracy: 0.9383886255924171
Test Accuracy: 0.9299999999999999

________________________________________

"""