import numpy as np
import matplotlib.pyplot as plt
import copy
import h5py
import time


""" Deep Feed Forward Neural Network (FFNN) for Boolean Classification Tasks"""


class DeepFeedForward(object):
    def __init__(
            self,
            hidden_activation="relu",
            output_activation="sigmoid",
            print_cost=False,
            visualize_cost=False,
            epsilon=100,
            sigma=1e-12,
            seed=42
    ):

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.print_cost = print_cost
        self.visualize_cost = visualize_cost
        self.epsilon = epsilon
        self.sigma = sigma
        self.seed = seed

        self.cost_ = []
        self.time_ = []
        self.costs = []
        self.times = []
        self.parameters = {}
        self.train_predict = None
        self.validation_predict = None
        self.test_predict = None

    def layer_sizes(self, X, Y, num_hidden_units):

        num_input_units = [X.shape[0]]
        num_output_units = [Y.shape[0]]
        num_hidden_units = list(num_hidden_units)

        layer_dimensions = list(num_input_units + num_hidden_units + num_output_units)

        return layer_dimensions

    def initialize_parameters(self, layer_dimensions):

        np.random.seed(self.seed)
        parameters = {}
        for l in range(1, len(layer_dimensions)):

            parameters[f"W{l}"] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) / np.sqrt(layer_dimensions[l-1])
            parameters[f"b{l}"] = np.zeros((layer_dimensions[l], 1))

        return parameters

    def sigmoid(self, Z):

        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    def relu(self, Z):

        A = np.maximum(0, Z)
        cache = Z

        return A, cache

    def sigmoid_backward(self, dA, cache):

        Z = cache
        S = 1 / (1 + np.exp(-Z))
        dZ = dA * S * (1 - S)

        assert (dZ.shape == Z.shape)

        return dZ

    def relu_backward(self, dA, cache):

        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    def linear_forward(self, A, W, b):

        Z = np.dot(W, A) + b

        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):

        if activation == "sigmoid":

            Z, linear_cache = self.linear_forward(A=A_prev, W=W, b=b)
            A, activation_cache = self.sigmoid(Z=Z)

        elif activation == "relu":

            Z, linear_cache = self.linear_forward(A=A_prev, W=W, b=b)
            A, activation_cache = self.relu(Z=Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_propagation(self, X, parameters, hidden_activation, output_activation):

        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A

            A, cache = self.linear_activation_forward(
                A_prev=A_prev,
                W=parameters[f"W{l}"],
                b=parameters[f"b{l}"],
                activation=hidden_activation
            )
            caches.append(cache)

        A, cache = self.linear_activation_forward(
            A_prev=A,
            W=parameters[f"W{L}"],
            b=parameters[f"b{L}"],
            activation=output_activation
        )
        caches.append(cache)

        return A, caches

    def compute_cost(self, A, Y, sigma):

        m = Y.shape[1]

        cost = - np.sum(np.multiply(Y, np.log(A+sigma)) + np.multiply(1 - Y, np.log(1 - A+sigma))) / m

        cost = np.squeeze(cost)

        return cost


    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):

        linear_cache, activation_cache = cache

        if activation == "relu":

            dZ = self.relu_backward(
                dA=dA,
                cache=activation_cache
            )
            dA_prev, dW, db = self.linear_backward(
                dZ=dZ,
                cache=linear_cache
            )

        elif activation == "sigmoid":

            dZ = self.sigmoid_backward(
                dA=dA,
                cache=activation_cache
            )
            dA_prev, dW, db = self.linear_backward(
                dZ=dZ,
                cache=linear_cache
            )

        return dA_prev, dW, db

    def backward_propagation(self, A, Y, caches, hidden_activation, output_activation):

        grads = {}
        L = len(caches)
        Y = Y.reshape(A.shape)

        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))

        current_cache = caches[-1]
        dA_prev, dW, db = self.linear_activation_backward(
            dA=dA,
            cache=current_cache,
            activation=output_activation
        )

        grads[f"dA{L-1}"] = dA_prev
        grads[f"dW{L}"] = dW
        grads[f"db{L}"] = db

        for l in reversed(range(L - 1)):

            current_cache = caches[l]

            dA_prev, dW, db = self.linear_activation_backward(
                dA=grads[f"dA{l+1}"],
                cache=current_cache,
                activation=hidden_activation
            )

            grads[f"dA{l}"] = dA_prev
            grads[f"dW{l+1}"] = dW
            grads[f"db{l+1}"] = db

        return grads

    def update_parameters(self, params, grads, learning_rate):

        parameters = copy.deepcopy(params)
        L = len(parameters) // 2

        for l in range(L):
            parameters[f"W{l+1}"] -= learning_rate * grads[f"dW{l+1}"]
            parameters[f"b{l+1}"] -= learning_rate * grads[f"db{l+1}"]

        return parameters

    def predict(self, X, parameters, hidden_activation, output_activation):

        m = X.shape[1]
        predicted = np.zeros((1, m))

        probabilities, caches = self.forward_propagation(
            X=X,
            parameters=parameters,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

        predicted[0, :] = (probabilities > 0.5)

        return predicted

    def accuracy(self, predicted, Y):
        m = Y.shape[1]
        accuracy = np.sum((predicted == Y) / m)
        return accuracy

    def train(self, X, Y, num_hidden_units, hidden_activation, output_activation, epochs, learning_rate):

        layer_dimensions = self.layer_sizes(X, Y, num_hidden_units)
        parameters = self.initialize_parameters(layer_dimensions)

        for i in range(epochs):

            start = time.time()

            A, caches = self.forward_propagation(
                X=X,
                parameters=parameters,
                hidden_activation=hidden_activation,
                output_activation=output_activation
            )

            cost = self.compute_cost(A=A, Y=Y, sigma=self.sigma)
            self.costs.append(cost)

            grads = self.backward_propagation(
                A=A,
                Y=Y,
                caches=caches,
                hidden_activation=hidden_activation,
                output_activation=output_activation
            )

            parameters = self.update_parameters(
                params=parameters,
                grads=grads,
                learning_rate=learning_rate
            )

            stop = time.time()
            duration = stop - start
            self.times.append(duration)

            if self.print_cost:
                if i % self.epsilon == 0 or i == epochs - 1:
                    print(f"Epoch: {i}; Cost: {cost}, Duration: {round(duration, 3)} S")
                    self.cost_.append(cost)
                    self.time_.append(duration)

        return parameters

    def nn_model(self, X_train, Y_train, num_hidden_units, X_validation=None, Y_validation=None, X_test=None,
                 Y_test=None, hidden_activation="relu", output_activation="sigmoid", epochs=1000, learning_rate=1e-4,
                 accuracy=None
                 ):
        if self.print_cost:
            print("___"*23)
            print(f"NN_Model with {len(num_hidden_units)+1} layers; Number of Epochs: {epochs}; Learning Rate: {learning_rate}")
            print("___"*23)

        parameters = self.train(
            X=X_train,
            Y=Y_train,
            num_hidden_units=num_hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            epochs=epochs,
            learning_rate=learning_rate
        )
        self.parameters = parameters

        if accuracy:
            print("__"*18)
            train_predict = self.predict(
                X=X_train,
                parameters=parameters,
                hidden_activation=hidden_activation,
                output_activation=output_activation
            )
            self.train_predict = train_predict
            train_accuracy = self.accuracy(
                predicted=train_predict,
                Y=Y_train
            )

            print(f"Train Accuracy: {train_accuracy}")

            if X_validation is not None and X_validation.any():

                validation_predict = self.predict(
                    X=X_validation,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation
                )
                self.validation_predict = validation_predict
                validation_accuracy = self.accuracy(
                    predicted=validation_predict,
                    Y=Y_validation
                )

                print(f"Validation Accuracy: {validation_accuracy}")

            if X_test is not None and X_test.any():
                test_predict = self.predict(
                    X=X_test,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation
                )
                self.test_predict = test_predict
                test_accuracy = self.accuracy(
                    predicted=test_predict,
                    Y=Y_test
                )

                print(f"Test Accuracy: {test_accuracy}")
                print("__" * 18)

            if self.visualize_cost:
                plt.figure(figsize=(8, 6))
                plt.plot(self.costs)
                plt.title("Epoch vs. Cost")
                plt.xlabel("Epoch")
                plt.ylabel("Cost")
                plt.show()




train_dataset = h5py.File("/home/samani/Documents/projects/deep-learning/data/train_cat.h5", "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

test_dataset = h5py.File("/home/samani/Documents/projects/deep-learning/data/test_cat.h5", "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

classes = np.array(test_dataset["list_classes"][:])  # the list of classes

train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print(train_set_x_flatten.shape)
print(test_set_x_flatten.shape)

X_train = train_set_x_flatten / 255
Y_train = train_set_y_orig
X_test = test_set_x_flatten / 255
Y_test = test_set_y_orig
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


model = DeepFeedForward(
    hidden_activation="relu",
    output_activation="sigmoid",
    print_cost=True,
    visualize_cost=True,
    epsilon=200,
    sigma=1e-12,
    seed=42
)

model.nn_model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[20, 10, 8, 4],
    X_validation=None,
    Y_validation=None,
    X_test=X_test,
    Y_test=Y_test,
    hidden_activation="relu",
    output_activation="sigmoid",
    epochs=4000,
    learning_rate=0.006,
    accuracy=True
)

"""
_____________________________________________________________________
NN_Model with 4 layers; Number of Epochs: 3000; Learning Rate: 0.001
_____________________________________________________________________
Epoch: 0; Cost: 0.7012713501337946, Duration: 0.04 S
Epoch: 100; Cost: 0.6880821484385041, Duration: 0.015 S
Epoch: 200; Cost: 0.6816900920720175, Duration: 0.016 S
Epoch: 300; Cost: 0.6704597866150025, Duration: 0.017 S
Epoch: 400; Cost: 0.6526195455216031, Duration: 0.015 S
Epoch: 500; Cost: 0.6371617464663251, Duration: 0.016 S
Epoch: 600; Cost: 0.6215167400483436, Duration: 0.015 S
Epoch: 700; Cost: 0.6065688539586512, Duration: 0.015 S
Epoch: 800; Cost: 0.59331313527278, Duration: 0.015 S
Epoch: 900; Cost: 0.5804205372651136, Duration: 0.015 S
Epoch: 1000; Cost: 0.5669278834461078, Duration: 0.015 S
Epoch: 1100; Cost: 0.5532368043746674, Duration: 0.015 S
Epoch: 1200; Cost: 0.5394531522889264, Duration: 0.015 S
Epoch: 1300; Cost: 0.5252370580519745, Duration: 0.015 S
Epoch: 1400; Cost: 0.5101793109401354, Duration: 0.015 S
Epoch: 1500; Cost: 0.4943434606260411, Duration: 0.015 S
Epoch: 1600; Cost: 0.4780696847164151, Duration: 0.015 S
Epoch: 1700; Cost: 0.4617743039895693, Duration: 0.015 S
Epoch: 1800; Cost: 0.44529979270383574, Duration: 0.018 S
Epoch: 1900; Cost: 0.428259253181542, Duration: 0.015 S
Epoch: 2000; Cost: 0.4106890037449624, Duration: 0.016 S
Epoch: 2100; Cost: 0.39303215862240504, Duration: 0.021 S
Epoch: 2200; Cost: 0.37521276145514576, Duration: 0.015 S
Epoch: 2300; Cost: 0.3575271780517624, Duration: 0.015 S
Epoch: 2400; Cost: 0.33990626095526083, Duration: 0.017 S
Epoch: 2500; Cost: 0.3219000922209901, Duration: 0.015 S
Epoch: 2600; Cost: 0.3040539243633297, Duration: 0.015 S
Epoch: 2700; Cost: 0.2866286357479098, Duration: 0.015 S
Epoch: 2800; Cost: 0.26972719276799406, Duration: 0.022 S
Epoch: 2900; Cost: 0.2533136053074878, Duration: 0.015 S
Epoch: 2999; Cost: 0.23763882624274776, Duration: 0.015 S
____________________________________
Train Accuracy: 0.9425837320574161
Test Accuracy: 0.74
____________________________________
"""

"""
_____________________________________________________________________
NN_Model with 4 layers; Number of Epochs: 4000; Learning Rate: 0.001
_____________________________________________________________________
Epoch: 0; Cost: 0.7012713501337946, Duration: 0.019 S
Epoch: 200; Cost: 0.6816900920720175, Duration: 0.016 S
Epoch: 400; Cost: 0.6526195455216031, Duration: 0.018 S
Epoch: 600; Cost: 0.6215167400483436, Duration: 0.015 S
Epoch: 800; Cost: 0.59331313527278, Duration: 0.015 S
Epoch: 1000; Cost: 0.5669278834461078, Duration: 0.015 S
Epoch: 1200; Cost: 0.5394531522889264, Duration: 0.015 S
Epoch: 1400; Cost: 0.5101793109401354, Duration: 0.016 S
Epoch: 1600; Cost: 0.4780696847164151, Duration: 0.018 S
Epoch: 1800; Cost: 0.44529979270383574, Duration: 0.016 S
Epoch: 2000; Cost: 0.4106890037449624, Duration: 0.017 S
Epoch: 2200; Cost: 0.37521276145514576, Duration: 0.015 S
Epoch: 2400; Cost: 0.33990626095526083, Duration: 0.015 S
Epoch: 2600; Cost: 0.3040539243633297, Duration: 0.015 S
Epoch: 2800; Cost: 0.26972719276799406, Duration: 0.015 S
Epoch: 3000; Cost: 0.23745627570622083, Duration: 0.019 S
Epoch: 3200; Cost: 0.20787057709443862, Duration: 0.015 S
Epoch: 3400; Cost: 0.18125360453985917, Duration: 0.015 S
Epoch: 3600; Cost: 0.15762738089363942, Duration: 0.015 S
Epoch: 3800; Cost: 0.1369431591760364, Duration: 0.015 S
Epoch: 3999; Cost: 0.11896178829744164, Duration: 0.018 S
____________________________________
Train Accuracy: 0.9952153110047844
Test Accuracy: 0.76
____________________________________

"""

"""
_____________________________________________________________________
NN_Model with 4 layers; Number of Epochs: 5000; Learning Rate: 0.001
_____________________________________________________________________
Epoch: 0; Cost: 0.7012713501337946, Duration: 0.021 S
Epoch: 200; Cost: 0.6816900920720175, Duration: 0.02 S
Epoch: 400; Cost: 0.6526195455216031, Duration: 0.015 S
Epoch: 600; Cost: 0.6215167400483436, Duration: 0.015 S
Epoch: 800; Cost: 0.59331313527278, Duration: 0.021 S
Epoch: 1000; Cost: 0.5669278834461078, Duration: 0.015 S
Epoch: 1200; Cost: 0.5394531522889264, Duration: 0.017 S
Epoch: 1400; Cost: 0.5101793109401354, Duration: 0.015 S
Epoch: 1600; Cost: 0.4780696847164151, Duration: 0.015 S
Epoch: 1800; Cost: 0.44529979270383574, Duration: 0.015 S
Epoch: 2000; Cost: 0.4106890037449624, Duration: 0.015 S
Epoch: 2200; Cost: 0.37521276145514576, Duration: 0.015 S
Epoch: 2400; Cost: 0.33990626095526083, Duration: 0.015 S
Epoch: 2600; Cost: 0.3040539243633297, Duration: 0.015 S
Epoch: 2800; Cost: 0.26972719276799406, Duration: 0.016 S
Epoch: 3000; Cost: 0.23745627570622083, Duration: 0.015 S
Epoch: 3200; Cost: 0.20787057709443862, Duration: 0.015 S
Epoch: 3400; Cost: 0.18125360453985917, Duration: 0.015 S
Epoch: 3600; Cost: 0.15762738089363942, Duration: 0.015 S
Epoch: 3800; Cost: 0.1369431591760364, Duration: 0.016 S
Epoch: 4000; Cost: 0.11888378422118798, Duration: 0.015 S
Epoch: 4200; Cost: 0.10363555890761972, Duration: 0.015 S
Epoch: 4400; Cost: 0.09067701049630374, Duration: 0.015 S
Epoch: 4600; Cost: 0.07963142743306308, Duration: 0.015 S
Epoch: 4800; Cost: 0.07024612013436367, Duration: 0.015 S
Epoch: 4999; Cost: 0.06231086262186377, Duration: 0.015 S
____________________________________
Train Accuracy: 0.9999999999999998
Test Accuracy: 0.76
____________________________________
"""

"""
_____________________________________________________________________
NN_Model with 5 layers; Number of Epochs: 4000; Learning Rate: 0.006
_____________________________________________________________________
Epoch: 0; Cost: 0.693330159998475, Duration: 0.028 S
Epoch: 200; Cost: 0.6663451382568525, Duration: 0.018 S
Epoch: 400; Cost: 0.6460935202840066, Duration: 0.018 S
Epoch: 600; Cost: 0.6162170719339137, Duration: 0.022 S
Epoch: 800; Cost: 0.5548425122209394, Duration: 0.025 S
Epoch: 1000; Cost: 0.4802680098261366, Duration: 0.021 S
Epoch: 1200; Cost: 0.4029390646231326, Duration: 0.035 S
Epoch: 1400; Cost: 0.3321133853304996, Duration: 0.07 S
Epoch: 1600; Cost: 0.25336799482689026, Duration: 0.039 S
Epoch: 1800; Cost: 0.20970083773703457, Duration: 0.018 S
Epoch: 2000; Cost: 0.1830097228425602, Duration: 0.033 S
Epoch: 2200; Cost: 0.16518057542044307, Duration: 0.019 S
Epoch: 2400; Cost: 0.1506133800991855, Duration: 0.064 S
Epoch: 2600; Cost: 0.13974109436163507, Duration: 0.028 S
Epoch: 2800; Cost: 0.1316378800675738, Duration: 0.019 S
Epoch: 3000; Cost: 0.12333962559285484, Duration: 0.038 S
Epoch: 3200; Cost: 0.11725373231501997, Duration: 0.019 S
Epoch: 3400; Cost: 0.11210206203991768, Duration: 0.021 S
Epoch: 3600; Cost: 0.10801038159607114, Duration: 0.018 S
Epoch: 3800; Cost: 0.10402645362808875, Duration: 0.021 S
Epoch: 3999; Cost: 0.10074731290173176, Duration: 0.035 S
____________________________________
Train Accuracy: 0.9856459330143538
Test Accuracy: 0.8
____________________________________
"""




















