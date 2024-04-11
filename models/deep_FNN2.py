import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import copy
import time

""" Deep Feed Forward Neural Network (FFNN) with Different Parameter Initialization Methods for Boolean Classification Tasks"""


class DeepFeedForward(object):
    def __init__(self, hidden_activation="relu", output_activation="sigmoid", print_cost=False, visualize_cost=False, epsilon=1000, sigma=1e-12, seed=None):

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

    # implement parameter initialization function using He method
    def initialize_parameters_he(self, num_layer_units, seed):

        if seed:
            np.random.seed(seed)
        parameters = {}

        for l in range(1, len(num_layer_units)):
            parameters[f"W{l}"] = np.random.randn(num_layer_units[l], num_layer_units[l - 1]) * np.sqrt(2 / num_layer_units[l - 1])
            parameters[f"b{l}"] = np.zeros((num_layer_units[l], 1))

        return parameters

    def initialize_parameters_deep(self, num_layer_units, seed):

        if seed:
            np.random.seed(seed)

        parameters = {}

        for l in range(1, len(num_layer_units)):
            parameters[f"W{l}"] = np.random.randn(num_layer_units[l], num_layer_units[l - 1]) / np.sqrt(num_layer_units[l - 1])
            parameters[f"b{l}"] = np.zeros((num_layer_units[l], 1))

        return parameters

    # implement parameter initialization function using Xavier method
    def initialize_parameters_xavier(self, num_layer_units, seed):

        if seed:
            np.random.seed(seed)

        parameters = {}

        for l in range(1, len(num_layer_units)):
            parameters[f"W{l}"] = np.random.randn(num_layer_units[l], num_layer_units[l - 1]) * np.sqrt(1 / num_layer_units[l - 1])
            parameters[f"b{l}"] = np.zeros((num_layer_units[l], 1))

        return parameters

    # implement parameter initialization function using small gaussian random variables
    def initialize_parameters_random1(self, num_layer_units, seed):

        if seed:
            np.random.seed(seed)

        parameters = {}

        for l in range(1, len(num_layer_units)):
            parameters[f"W{l}"] = np.random.randn(num_layer_units[l], num_layer_units[l - 1]) * 0.01
            parameters[f"b{l}"] = np.zeros((num_layer_units[l], 1))

        return parameters

    # implement parameter initialization function using small gaussian random variables for both W and b
    def initialize_parameters_random2(self, num_layer_units, seed):

        if seed:
            np.random.seed(seed)

        parameters = {}

        for l in range(1, len(num_layer_units)):
            parameters[f"W{l}"] = np.random.randn(num_layer_units[l], num_layer_units[l - 1]) * 0.01
            parameters[f"b{l}"] = np.random.randn(num_layer_units[l], 1) * 0.01

        return parameters

    # implement parameter initialization function using small uniform random variables
    def initialize_parameters_random3(self, num_layer_units, seed):

        if seed:
            np.random.seed(seed)

        parameters = {}

        for l in range(1, len(num_layer_units)):
            parameters[f"W{l}"] = np.random.rand(num_layer_units[l], num_layer_units[l - 1]) * 0.01
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

    def helper_forward1(self, A_prev, W, b):

        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)

        return Z, cache

    def helper_forward2(self, A_prev, W, b, activation):

        if activation == "relu":

            Z, linear_cache = self.helper_forward1(A_prev=A_prev, W=W, b=b)
            A, activation_cache = self.relu(Z=Z)
        elif activation == "sigmoid":

            Z, linear_cache = self.helper_forward1(A_prev=A_prev, W=W, b=b)
            A, activation_cache = self.sigmoid(Z=Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_propagation(self, X, parameters, hidden_activation, output_activation):

        caches = []
        A = X
        num_layers = len(parameters) // 2

        for l in range(1, num_layers):
            A_prev = A
            A, cache = self.helper_forward2(
                A_prev=A_prev,
                W=parameters[f"W{l}"],
                b=parameters[f"b{l}"],
                activation=hidden_activation
            )
            caches.append(cache)

        A, cache = self.helper_forward2(
            A_prev=A,
            W=parameters[f"W{num_layers}"],
            b=parameters[f"b{num_layers}"],
            activation=output_activation
        )
        caches.append(cache)

        return A, caches

    def compute_cost(self, A, Y, sigma):

        m = Y.shape[1]

        cost = - np.sum(np.multiply(Y, np.log(A + sigma)) + np.multiply(1 - Y, np.log(1 - A + sigma))) / m
        cost = np.squeeze(cost)

        return cost

    def helper_backward1(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def helper_backward2(self, dA, cache, activation):

        linear_cache, activation_cache = cache

        if activation == "relu":

            dZ = self.derivative_relu(dA=dA, cache=activation_cache)
            dA_prev, dW, db = self.helper_backward1(dZ=dZ, cache=linear_cache)

        elif activation == "sigmoid":

            dZ = self.derivative_sigmoid(dA=dA, cache=activation_cache)
            dA_prev, dW, db = self.helper_backward1(dZ=dZ, cache=linear_cache)

        return dA_prev, dW, db

    def backward_propagation(self, A, Y, caches, hidden_activation, output_activation):

        grads = {}
        L = len(caches)
        Y = Y.reshape(A.shape)

        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))

        current_cache = caches[-1]
        dA_prev, dW, db = self.helper_backward2(dA=dA, cache=current_cache, activation=output_activation)
        grads[f"dW{L}"] = dW
        grads[f"db{L}"] = db
        grads[f"dA{L - 1}"] = dA_prev

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev, dW, db = self.helper_backward2(dA=grads[f"dA{l + 1}"], cache=current_cache,
                                               activation=hidden_activation)
            grads[f"dW{l + 1}"] = dW
            grads[f"db{l + 1}"] = db
            grads[f"dA{l}"] = dA_prev

        return grads

    def update_parameters(self, parameters, grads, learning_rate):

        parameters_ = copy.deepcopy(parameters)
        L = len(parameters_) // 2

        for l in range(L):
            parameters_[f"W{l + 1}"] -= learning_rate * grads[f"dW{l + 1}"]
            parameters_[f"b{l + 1}"] -= learning_rate * grads[f"db{l + 1}"]

        return parameters_

    def predict(self, X, parameters, hidden_activation, output_activation):

        predicted = np.zeros((1, X.shape[1]))

        probabilities, caches = self.forward_propagation(
            X=X,
            parameters=parameters,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

        predicted[0, :] = (probabilities > 0.5)

        return predicted

    def accuracy(self, predicted, Y):

        accuracy = np.sum((predicted == Y) / Y.shape[1])

        return accuracy

    def train(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, parameters,
              hidden_activation, output_activation, num_epochs, learning_rate, print_cost, epsilon, sigma):

        print()
        for i in range(1, num_epochs + 1):
            tic = time.time()

            A, caches = self.forward_propagation(
                X=X_train,
                parameters=parameters,
                hidden_activation=hidden_activation,
                output_activation=output_activation
            )
            cost = self.compute_cost(A=A, Y=Y_train, sigma=sigma)
            self.train_cost.append(cost)

            if X_validation is not None and X_validation.any():

                A_valid, _ = self.forward_propagation(
                    X=X_validation,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation
                )
                cost_v = self.compute_cost(A=A_valid, Y=Y_validation, sigma=sigma)
                self.validation_cost.append(cost_v)

            if X_test is not None and X_test.any():

                A_test, _ = self.forward_propagation(
                    X=X_test,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation
                )
                cost_t = self.compute_cost(A=A_test, Y=Y_test, sigma=sigma)
                self.test_cost.append(cost_t)

            grads = self.backward_propagation(
                A=A,
                Y=Y_train,
                caches=caches,
                hidden_activation=hidden_activation,
                output_activation=output_activation
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
                print(f"Epoch {i}; Cost: {cost}; Duration: {round(sum(self.time_[i - epsilon: i]), 5)} seconds")
            if i == num_epochs and i % epsilon != 0:
                print(f"Epoch {i}; Cost: {cost}; Duration: {round(toc - tic, 5)} seconds")

        return parameters

    def model(self, X_train, Y_train, num_hidden_units, X_validation=None, Y_validation=None, X_test=None, Y_test=None,
              param_init_method="he", num_epochs=1000, learning_rate=1e-3):

        num_layer_units = self.layer_sizes(X=X_train, Y=Y_train, num_hidden_units=num_hidden_units)

        if param_init_method == "he":
            parameters = self.initialize_parameters_he(num_layer_units=num_layer_units, seed=self.seed)
        elif param_init_method == "xavier":
            parameters = self.initialize_parameters_xavier(num_layer_units=num_layer_units, seed=self.seed)

        elif param_init_method == "gaussian_random 1":
            parameters = self.initialize_parameters_random1(num_layer_units=num_layer_units, seed=self.seed)
        elif param_init_method == "gaussian_random 2":
            parameters = self.initialize_parameters_random2(num_layer_units=num_layer_units, seed=self.seed)
        elif param_init_method == "uniform_random":
            parameters = self.initialize_parameters_random3(num_layer_units=num_layer_units, seed=self.seed)
        elif param_init_method == "deep":
            parameters = self.initialize_parameters_deep(num_layer_units=num_layer_units, seed=self.seed)

        if self.print_cost:
            print("###" * 45)
            print()
            print(f"NN_Model with {len(num_layer_units)} layers ({num_layer_units}); Number of Epochs: {num_epochs}; Learning Rate: {learning_rate}; Initialization Method: {param_init_method}")
            print()
            print("###" * 45)

        parameters_ = self.train(
            X_train=X_train,
            Y_train=Y_train,
            X_validation=X_validation,
            Y_validation=Y_validation,
            X_test=X_test,
            Y_test=Y_test,
            parameters=parameters,
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
            output_activation=self.output_activation
        )
        train_accuracy = self.accuracy(predicted=train_predicted, Y=Y_train)
        self.train_predict = train_predicted
        self.train_accuracy = train_accuracy

        if X_validation is not None and X_validation.any():
            valid_predicted = self.predict(
                X=X_validation,
                parameters=self.parameters,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation
            )
            valid_accuracy = self.accuracy(predicted=valid_predicted, Y=Y_validation)
            self.validation_predict = valid_predicted
            self.validation_accuracy = valid_accuracy

        if X_test is not None and X_test.any():
            test_predicted = self.predict(
                X=X_test,
                parameters=self.parameters,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation
            )
            test_accuracy = self.accuracy(predicted=test_predicted, Y=Y_test)
            self.test_predict = test_predicted
            self.test_accuracy = test_accuracy

        print("__" * 20)
        print()
        print(f"Train Accuracy: {self.train_accuracy}")
        if X_validation is not None and X_validation.any():
            print(f"Validation Accuracy: {self.validation_accuracy}")
        if X_test is not None and X_test.any():
            print(f"Test Accuracy: {self.test_accuracy}")
        print()
        print("__" * 20)

        if self.visualize_cost:
            plt.plot(self.train_cost, label="Train Cost")
            if X_validation is not None and X_validation.any():
                plt.plot(self.validation_cost, label="Validation Cost")
            if X_test is not None and X_test.any():
                plt.plot(self.test_cost, label="Test Cost")
            plt.title("Epoch vs. Cost")
            plt.xlabel(f"Epoch")
            plt.ylabel("Cost")
            plt.legend()
            plt.show()


N = 5000
np.random.seed(1)
noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                              n_classes=2, shuffle=True, random_state=None)
no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}


dataset = "gaussian_quantiles"


X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y%2

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.title(dataset)
plt.show()

print(X.shape)
print(Y.shape)
"""
(2, 5000)
(1, 5000)
"""
X_train = X[:, :4500]
Y_train = Y[:, :4500]
X_validation = X[:, 4500:4700]
Y_validation = Y[:, 4500:4700]
X_test = X[:, 4700:]
Y_test = Y[:, 4700:]

print(X_train.shape)
print(Y_train.shape)
print(X_validation.shape)
print(Y_validation.shape)
print(X_test.shape)
print(Y_test.shape)
"""
(2, 4500)
(1, 4500)
(2, 200)
(1, 200)
(2, 300)
(1, 300)
"""

model = DeepFeedForward(
    hidden_activation="relu",
    output_activation="sigmoid",
    print_cost=True,
    visualize_cost=True,
    epsilon=500,
    sigma=1e-12,
    seed=23
)


model.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[20, 10, 10, 8],
    X_validation=X_validation,
    Y_validation=Y_validation,
    X_test=X_test,
    Y_test=Y_test,
    param_init_method="he",
    num_epochs=7000,
    learning_rate=1e-3
)


"""
########################################################################################################################

NN_Model with 6 layers ([2, 20, 10, 10, 8, 1]); Number of Epochs: 7000; Learning Rate: 0.001; Initialization Method: he

########################################################################################################################

Epoch 500; Cost: 0.6559890916663041; Duration: 3.85795 seconds
Epoch 1000; Cost: 0.6441264101348229; Duration: 3.29262 seconds
Epoch 1500; Cost: 0.631161136893325; Duration: 3.27256 seconds
Epoch 2000; Cost: 0.6169066363077391; Duration: 3.29698 seconds
Epoch 2500; Cost: 0.60145624229137; Duration: 3.20478 seconds
Epoch 3000; Cost: 0.5846872171928422; Duration: 3.23754 seconds
Epoch 3500; Cost: 0.5659541336567403; Duration: 3.18288 seconds
Epoch 4000; Cost: 0.5427139805009744; Duration: 3.21957 seconds
Epoch 4500; Cost: 0.5185877296345777; Duration: 3.19535 seconds
Epoch 5000; Cost: 0.4924250529468497; Duration: 3.21502 seconds
Epoch 5500; Cost: 0.463986822563764; Duration: 3.19844 seconds
Epoch 6000; Cost: 0.43637813263612196; Duration: 3.19518 seconds
Epoch 6500; Cost: 0.40944605081483343; Duration: 3.22312 seconds
Epoch 7000; Cost: 0.38342551562000254; Duration: 3.27283 seconds

________________________________________

Train Accuracy: 0.9113333333333333
Validation Accuracy: 0.8899999999999999
Test Accuracy: 0.9100000000000001

________________________________________
"""

model.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[20, 10, 10, 8],
    X_validation=X_validation,
    Y_validation=Y_validation,
    X_test=X_test,
    Y_test=Y_test,
    param_init_method="he",
    num_epochs=10000,
    learning_rate=1e-3
)

"""
########################################################################################################################

NN_Model with 6 layers ([2, 20, 10, 10, 8, 1]); Number of Epochs: 10000; Learning Rate: 0.001; Initialization Method: he

########################################################################################################################

Epoch 500; Cost: 0.6559890916663041; Duration: 4.09061 seconds
Epoch 1000; Cost: 0.6441264101348229; Duration: 3.23843 seconds
Epoch 1500; Cost: 0.631161136893325; Duration: 3.20849 seconds
Epoch 2000; Cost: 0.6169066363077391; Duration: 3.21538 seconds
Epoch 2500; Cost: 0.60145624229137; Duration: 3.27814 seconds
Epoch 3000; Cost: 0.5846872171928422; Duration: 3.23746 seconds
Epoch 3500; Cost: 0.5659541336567403; Duration: 3.55845 seconds
Epoch 4000; Cost: 0.5427139805009744; Duration: 3.29821 seconds
Epoch 4500; Cost: 0.5185877296345777; Duration: 3.24655 seconds
Epoch 5000; Cost: 0.4924250529468497; Duration: 3.1936 seconds
Epoch 5500; Cost: 0.463986822563764; Duration: 3.47196 seconds
Epoch 6000; Cost: 0.43637813263612196; Duration: 3.20373 seconds
Epoch 6500; Cost: 0.40944605081483343; Duration: 3.53913 seconds
Epoch 7000; Cost: 0.38342551562000254; Duration: 3.1734 seconds
Epoch 7500; Cost: 0.35829946400736423; Duration: 3.25221 seconds
Epoch 8000; Cost: 0.3340228224920717; Duration: 3.16801 seconds
Epoch 8500; Cost: 0.3105998906970439; Duration: 3.19205 seconds
Epoch 9000; Cost: 0.28788774041768533; Duration: 3.18804 seconds
Epoch 9500; Cost: 0.2652476896177351; Duration: 3.26742 seconds
Epoch 10000; Cost: 0.2430740667545965; Duration: 3.13697 seconds

________________________________________

Train Accuracy: 0.965111111111111
Validation Accuracy: 0.9449999999999998
Test Accuracy: 0.9500000000000002

________________________________________
"""


