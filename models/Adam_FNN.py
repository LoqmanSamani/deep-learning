import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import scipy.io
import time
import h5py

""" Implementation of Feed Forward Neural Network With Adam Algorithm and Mini-Batch/Stochastic Technique"""


class AdamDFFNN:
    def __init__(self, print_cost=False, visualize_cost=False, gamma=100, epsilon=1e-8, decay_rate=0.3, time_interval=100, minibatch_size=64, bias_correction=False, seed=1):

        self.print_cost = print_cost
        self.visualize_cost = visualize_cost
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.time_interval = time_interval
        self.minibatch_size = minibatch_size
        self.bias_correction = bias_correction
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
        num_hidden_units = list(num_hidden_units)
        num_output_units = [Y.shape[0]]
        layer_units = list(num_input_units + num_hidden_units + num_output_units)

        return layer_units

    def initialize_parameters(self, layer_units, seed):

        np.random.seed(seed)

        parameters = {}
        L = len(layer_units)
        for l in range(1, L):

            parameters[f"W{l}"] = np.random.randn(layer_units[l], layer_units[l - 1]) * np.sqrt(2 / layer_units[l - 1])
            parameters[f"b{l}"] = np.zeros((layer_units[l], 1))

        return parameters

    def initialize_adam_parameters(self, parameters):

        momentum = {}
        squared_gradients = {}
        L = len(parameters) // 2

        for l in range(1, L + 1):

            momentum[f"vdW{l}"] = np.zeros(parameters[f"W{l}"].shape)
            momentum[f"vdb{l}"] = np.zeros(parameters[f"b{l}"].shape)
            squared_gradients[f"sdW{l}"] = np.zeros(parameters[f"W{l}"].shape)
            squared_gradients[f"sdb{l}"] = np.zeros(parameters[f"b{l}"].shape)

        return (momentum, squared_gradients)

    def relu(self, Z):
        A = np.maximum(0, Z)
        cache = Z

        return A, cache

    def sigmoid(self, Z):

        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    def backward_sigmoid(self, dA, Z):

        A = 1 / (1 + np.exp(-Z))
        dZ = np.multiply(np.multiply(dA, A), 1 - A)
        assert (dZ.shape == Z.shape)

        return dZ

    def backward_relu(self, dA, Z):

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)

        return dZ

    def forward_propagation(self, X, parameters):

        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A
            W = parameters[f"W{l}"]
            b = parameters[f"b{l}"]

            linear_cache = (A_prev, W, b)
            Z = np.dot(W, A_prev) + b
            A, activation_cache = self.relu(Z=Z)
            caches.append((linear_cache, activation_cache))

        A_prev = A
        W = parameters[f"W{L}"]
        b = parameters[f"b{L}"]
        linear_cache = (A_prev, W, b)

        Z = np.dot(W, A_prev) + b
        A, activation_cache = self.sigmoid(Z=Z)

        caches.append((linear_cache, activation_cache))

        return A, caches

    def compute_cost(self, A, Y):
        """
        This is used with mini-batches,
        so we'll first accumulate costs over an entire epoch
        and then divide by the m training examples
        """
        log_probs = np.multiply(-np.log(A + 1e-15), Y) + np.multiply(-np.log(1 - A + 1e-15), 1 - Y)
        cost_total = np.sum(log_probs)

        return cost_total

    def backward_propagation(self, AL, Y, caches):

        Y = Y.reshape(AL.shape)
        L = len(caches)
        gradients = {}

        dAL = - (np.divide(Y, AL + 1e-15) - np.divide(1 - Y, 1 - AL + 1e-15))

        (linear_cache, activation_cache) = caches[-1]
        (A_prev, W, b) = linear_cache
        m = A_prev.shape[1]

        dZ = self.backward_sigmoid(dA=dAL, Z=activation_cache)
        gradients[f"dA{L - 1}"] = np.dot(W.T, dZ)
        gradients[f"dW{L}"] = 1 / m * np.dot(dZ, A_prev.T)
        gradients[f"db{L}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        for l in reversed(range(L - 1)):
            (linear_cache, activation_cache) = caches[l]
            (A_prev, W, b) = linear_cache
            m = A_prev.shape[1]

            dZ = self.backward_relu(dA=gradients[f"dA{l + 1}"], Z=activation_cache)
            gradients[f"dA{l}"] = np.dot(W.T, dZ)
            gradients[f"dW{l + 1}"] = 1 / m * np.dot(dZ, A_prev.T)
            gradients[f"db{l + 1}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        return gradients

    def compute_momentum(self, parameters, gradients, momentum, beta, bias_correction, t):

        L = len(parameters) // 2

        for l in range(1, L + 1):

            momentum[f"vdW{l}"] = (beta * momentum[f"vdW{l}"]) + ((1 - beta) * gradients[f"dW{l}"])
            momentum[f"vdb{l}"] = (beta * momentum[f"vdb{l}"]) + ((1 - beta) * gradients[f"db{l}"])

            # bias correction terms
            if bias_correction:
                momentum[f"vdW{l}"] = momentum[f"vdW{l}"] / (1 - np.power(beta, t))
                momentum[f"vdb{l}"] = momentum[f"vdb{l}"] / (1 - np.power(beta, t))

        return momentum

    def compute_squared_gradients(self, parameters, gradients, squared_gradients, beta, bias_correction, t):

        L = len(parameters) // 2

        for l in range(1, L + 1):

            squared_gradients[f"sdW{l}"] = (beta * squared_gradients[f"sdW{l}"]) + ((1 - beta) * np.power(gradients[f"dW{l}"], 2))
            squared_gradients[f"sdb{l}"] = (beta * squared_gradients[f"sdb{l}"]) + ((1 - beta) * np.power(gradients[f"db{l}"], 2))

            # bias correction terms
            if bias_correction:
                squared_gradients[f"sdW{l}"] = squared_gradients[f"sdW{l}"] / (1 - np.power(beta, t))
                squared_gradients[f"sdb{l}"] = squared_gradients[f"sdb{l}"] / (1 - np.power(beta, t))

        return squared_gradients

    def update_parameters(self, parameters, momentum, squared_gradients, learning_rate, epsilon):

        L = len(parameters) // 2

        for l in range(1, L + 1):

            parameters[f"W{l}"] -= learning_rate * (momentum[f"vdW{l}"] / np.sqrt(squared_gradients[f"sdW{l}"] + epsilon))
            parameters[f"b{l}"] -= learning_rate * (momentum[f"vdb{l}"] / np.sqrt(squared_gradients[f"sdb{l}"] + epsilon))

        return parameters

    def update_learning_rate(self, init_learning_rate, num_epoch, decay_rate, time_interval):

        learning_rate = (1 * init_learning_rate) / (1 + decay_rate * (np.floor(num_epoch / time_interval)))

        return learning_rate

    def predict(self, X, parameters):

        A, _ = self.forward_propagation(X=X, parameters=parameters)

        A = (A > 0.5).astype(int)

        return A

    def accuracy(self, A, Y):

        accuracy = np.sum((A == Y) / Y.shape[1])

        return accuracy

    def random_mini_batches(self, X, Y, mini_batch_size, seed):

        if seed:
            np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []

        permutation = np.random.permutation(m)
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        # Step 2: Partition the shuffled data into mini-batches
        num_complete_minibatches = m // mini_batch_size

        for k in range(num_complete_minibatches):
            start_idx = k * mini_batch_size
            end_idx = (k + 1) * mini_batch_size
            mini_batch_X = shuffled_X[:, start_idx:end_idx]
            mini_batch_Y = shuffled_Y[:, start_idx:end_idx]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        # Handling the end case (last mini-batch may have fewer examples)
        if m % mini_batch_size != 0:
            start_idx = num_complete_minibatches * mini_batch_size
            mini_batch_X = shuffled_X[:, start_idx:]
            mini_batch_Y = shuffled_Y[:, start_idx:]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        return mini_batches

    def train(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, parameters, momentum, squared_gradients,  num_epochs, learning_rate, beta1, beta2):

        seed = int(self.seed)
        init_learning_rate = learning_rate
        epoch_num = 0
        m = X_train.shape[1]
        t = 0
        print()
        for i in range(num_epochs):
            tic = time.time()

            seed += 1
            mini_batches = self.random_mini_batches(
                X=X_train,
                Y=Y_train,
                mini_batch_size=self.minibatch_size,
                seed=seed
            )
            cost_total = 0
            for minibatch in mini_batches:
                (X_minibatch, Y_minibatch) = minibatch
                A_minibatch, minibatch_caches = self.forward_propagation(
                    X=X_minibatch,
                    parameters=parameters
                )
                cost_total += self.compute_cost(
                    A=A_minibatch,
                    Y=Y_minibatch
                )

                gradients = self.backward_propagation(
                    AL=A_minibatch,
                    Y=Y_minibatch,
                    caches=minibatch_caches
                )
                t += 1
                momentum = self.compute_momentum(parameters=parameters,
                                                 gradients=gradients,
                                                 momentum=momentum,
                                                 beta=beta1,
                                                 bias_correction=self.bias_correction,
                                                 t=t
                                                 )
                squared_gradients = self.compute_squared_gradients(
                    parameters=parameters,
                    gradients=gradients,
                    squared_gradients=squared_gradients,
                    beta=beta2,
                    bias_correction=self.bias_correction,
                    t=t
                )
                parameters = self.update_parameters(
                    parameters=parameters,
                    momentum=momentum,
                    squared_gradients=squared_gradients,
                    learning_rate=learning_rate,
                    epsilon=self.epsilon
                )
                self.train_cost.append(cost_total / m)
                epoch_num += 1

                learning_rate = self.update_learning_rate(
                    init_learning_rate=init_learning_rate,
                    num_epoch=num_epochs,
                    decay_rate=self.decay_rate,
                    time_interval=self.time_interval
                )

            if X_validation is not None:

                A_valid, _ = self.forward_propagation(
                    X=X_validation,
                    parameters=parameters
                )

                cost_v = self.compute_cost(
                    A=A_valid,
                    Y=Y_validation
                )
                self.validation_cost.append(cost_v / X_validation.shape[1])

            if X_test is not None:

                A_test, _ = self.forward_propagation(
                    X=X_test,
                    parameters=parameters,
                )

                cost_t = self.compute_cost(
                    A=A_test,
                    Y=Y_test
                )
                self.test_cost.append(cost_t / X_test.shape[1])

            self.parameters_[i] = parameters
            toc = time.time()
            self.time_.append(toc - tic)
            if self.print_cost and i % self.gamma == 0:
                print(f"Epoch {i}; Cost: {round(cost_total / m, 5)}; Duration: {round(sum(self.time_[i - self.gamma: i]), 5)} seconds")

        return parameters

    def model(self, X_train, Y_train, num_hidden_units, X_validation=None, Y_validation=None, X_test=None, Y_test=None, num_epochs=1000, learning_rate=1e-4, beta1=0.9, beta2=0.99):

        num_layer_units = self.layer_sizes(
            X=X_train,
            Y=Y_train,
            num_hidden_units=num_hidden_units
        )
        parameters = self.initialize_parameters(
            layer_units=num_layer_units,
            seed=self.seed
        )
        (momentum, squared_gradients) = self.initialize_adam_parameters(parameters=parameters)
        if self.print_cost:
            print("___" * 35)
            print()
            print(f"NN_Model with {len(num_hidden_units) + 1} layers ({num_layer_units}); Number of Epochs: {num_epochs}; Learning Rate: {learning_rate}")
            print()
            print("___" * 35)

        parameters_ = self.train(
            X_train=X_train,
            Y_train=Y_train,
            X_validation=X_validation,
            Y_validation=Y_validation,
            X_test=X_test,
            Y_test=Y_test,
            parameters=parameters,
            momentum=momentum,
            squared_gradients=squared_gradients,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2
        )
        print()
        self.parameters = parameters_

        train_predicted = self.predict(
            X=X_train,
            parameters=self.parameters
        )
        train_accuracy = self.accuracy(
            A=train_predicted,
            Y=Y_train
        )
        self.train_predict = train_predicted
        self.train_accuracy = train_accuracy

        if X_validation is not None:
            valid_predicted = self.predict(
                X=X_validation,
                parameters=self.parameters
            )
            valid_accuracy = self.accuracy(
                A=valid_predicted,
                Y=Y_validation
            )
            self.validation_predict = valid_predicted
            self.validation_accuracy = valid_accuracy

        if X_test is not None:
            test_predicted = self.predict(
                X=X_test,
                parameters=self.parameters
            )
            test_accuracy = self.accuracy(
                A=test_predicted,
                Y=Y_test
            )
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

obj = AdamDFFNN(
    print_cost=True,
    visualize_cost=True,
    gamma=1000,
    epsilon=1e-8,
    decay_rate=0.3,
    time_interval=100,
    minibatch_size=32,
    bias_correction=False,
    seed=1
)


obj.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[10, 6, 4],
    X_test=X_test,
    Y_test=Y_test,
    num_epochs=15000,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.99
)


"""
__________________________________________________________________________________________

NN_Model with 4 layers ([2, 10, 6, 4, 1]); Number of Epochs: 15000; Learning Rate: 0.001

___________________________________________________________________________________________

Epoch 0; Cost: 0.68358; Duration: 0 seconds
Epoch 1000; Cost: 0.53866; Duration: 3.25561 seconds
Epoch 2000; Cost: 0.44901; Duration: 3.07315 seconds
Epoch 3000; Cost: 0.4017; Duration: 3.07778 seconds
Epoch 4000; Cost: 0.37223; Duration: 3.10245 seconds
Epoch 5000; Cost: 0.34914; Duration: 3.07239 seconds
Epoch 6000; Cost: 0.33049; Duration: 3.08986 seconds
Epoch 7000; Cost: 0.3142; Duration: 3.09927 seconds
Epoch 8000; Cost: 0.29977; Duration: 3.09361 seconds
Epoch 9000; Cost: 0.28795; Duration: 3.51042 seconds
Epoch 10000; Cost: 0.27693; Duration: 3.27723 seconds
Epoch 11000; Cost: 0.26738; Duration: 3.15334 seconds
Epoch 12000; Cost: 0.25827; Duration: 3.14387 seconds
Epoch 13000; Cost: 0.25028; Duration: 3.15321 seconds
Epoch 14000; Cost: 0.24371; Duration: 3.16525 seconds

________________________________________

Train Accuracy: 0.933649289099526
Test Accuracy: 0.9249999999999998

________________________________________
"""

train_dataset = h5py.File("/home/samani/Documents/projects/deep-learning/data/train_cat.h5", "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File("/home/samani/Documents/projects/deep-learning/data/test_cat.h5", "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])

classes = np.array(test_dataset["list_classes"][:])

train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

X_train1 = train_set_x_flatten / 255
Y_train1 = train_set_y_orig
X_test1 = test_set_x_flatten / 255
Y_test1 = test_set_y_orig

obj1 = AdamDFFNN(
    print_cost=True,
    visualize_cost=True,
    gamma=200,
    epsilon=1e-8,
    decay_rate=0.3,
    time_interval=500,
    minibatch_size=64,
    bias_correction=False,
    seed=1
)
obj1.model(
    X_train=X_train1,
    Y_train=Y_train1,
    num_hidden_units=[10, 6],
    X_test=X_test1,
    Y_test=Y_test1,
    num_epochs=3000,
    learning_rate=1e-5,
    beta1=0.9,
    beta2=0.9
)

"""
__________________________________________________________________________________________

NN_Model with 3 layers ([12288, 10, 6, 1]); Number of Epochs: 3000; Learning Rate: 1e-05

__________________________________________________________________________________________

Epoch 0; Cost: 0.7451; Duration: 0 seconds
Epoch 200; Cost: 0.53713; Duration: 9.69402 seconds
Epoch 400; Cost: 0.45937; Duration: 9.51303 seconds
Epoch 600; Cost: 0.40251; Duration: 9.34752 seconds
Epoch 800; Cost: 0.35481; Duration: 9.33364 seconds
Epoch 1000; Cost: 0.31471; Duration: 9.35984 seconds
Epoch 1200; Cost: 0.28051; Duration: 9.36376 seconds
Epoch 1400; Cost: 0.25003; Duration: 9.31528 seconds
Epoch 1600; Cost: 0.22317; Duration: 9.30627 seconds
Epoch 1800; Cost: 0.19823; Duration: 9.55466 seconds
Epoch 2000; Cost: 0.15311; Duration: 9.37744 seconds
Epoch 2200; Cost: 0.12582; Duration: 9.36543 seconds
Epoch 2400; Cost: 0.10735; Duration: 9.32031 seconds
Epoch 2600; Cost: 0.09212; Duration: 9.39033 seconds
Epoch 2800; Cost: 0.07952; Duration: 9.3871 seconds

________________________________________

Train Accuracy: 0.9952153110047844
Test Accuracy: 0.78

________________________________________

"""

obj2 = AdamDFFNN(
    print_cost=True,
    visualize_cost=True,
    gamma=200,
    epsilon=1e-8,
    decay_rate=0.3,
    time_interval=500,
    minibatch_size=64,
    bias_correction=False,
    seed=1
)
obj2.model(
    X_train=X_train1,
    Y_train=Y_train1,
    num_hidden_units=[10, 6],
    X_test=X_test1,
    Y_test=Y_test1,
    num_epochs=2000,
    learning_rate=1e-5,
    beta1=0.9,
    beta2=0.9
)

"""
_________________________________________________________________________________________

NN_Model with 3 layers ([12288, 10, 6, 1]); Number of Epochs: 2000; Learning Rate: 1e-05

__________________________________________________________________________________________

Epoch 0; Cost: 0.74461; Duration: 0 seconds
Epoch 200; Cost: 0.51136; Duration: 11.76011 seconds
Epoch 400; Cost: 0.42711; Duration: 11.06641 seconds
Epoch 600; Cost: 0.36427; Duration: 9.89041 seconds
Epoch 800; Cost: 0.3115; Duration: 10.16286 seconds
Epoch 1000; Cost: 0.26926; Duration: 9.33059 seconds
Epoch 1200; Cost: 0.23346; Duration: 9.26339 seconds
Epoch 1400; Cost: 0.20134; Duration: 9.2687 seconds
Epoch 1600; Cost: 0.1705; Duration: 9.41049 seconds
Epoch 1800; Cost: 0.11844; Duration: 9.2986 seconds

________________________________________

Train Accuracy: 0.9999999999999998
Test Accuracy: 0.78

________________________________________

"""