import numpy as np
import matplotlib.pyplot as plt
import time
import h5py


"""Implementation of Deep Feed Forward Neural Network (DFFNN) with Adam Algorithm for Multiclass Classification Tasks with Learning Rate Decay and L2 Regularization"""


class MulticlassClassification:
    def __init__(self, print_cost=False, visualize_cost=False, num_print_cost=100, epsilon=1e-8, decay_rate=0.3, time_interval=100, minibatch_size=64, bias_correction=False, seed=1):

        self.print_cost = print_cost
        self.visualize_cost = visualize_cost
        self.num_print_cost = num_print_cost
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.time_interval = time_interval
        self.minibatch_size = minibatch_size
        self.bias_correction = bias_correction
        self.seed = seed

        self.train_cost = []
        self.test_cost = []
        self.time_ = None
        self.costs = []
        self.parameters_ = {}  # a dictionary for storing intermediate parameters
        self.parameters = {}  # a dictionary for storing final parameters
        self.learning_rates = []
        self.train_accuracy = []
        self.test_accuracy = []

    def onehot_encoding(self, Y):

        num_classes = len(np.unique(Y))
        one_hot_Y = np.eye(num_classes)[Y].T

        return one_hot_Y

    def layer_sizes(self, X, Y, num_hidden_units):

        num_input_units = X.shape[0]
        num_output_units = Y.shape[0]
        num_layer_units = [num_input_units] + list(num_hidden_units) + [num_output_units]

        return num_layer_units

    def initialize_parameters(self, num_layer_units):
        parameters = {}
        L = len(num_layer_units)

        for l in range(1, L):
            parameters[f"W{l}"] = np.random.randn(num_layer_units[l], num_layer_units[l - 1]) * np.sqrt(2 / num_layer_units[l - 1])
            parameters[f"b{l}"] = np.zeros((num_layer_units[l], 1))

        return parameters

    def initialize_adam_parameters(self, parameters):

        V = {}
        S = {}
        L = len(parameters) // 2

        for l in range(1, L + 1):
            V[f"vdW{l}"] = np.zeros(parameters[f"W{l}"].shape)
            V[f"vdb{l}"] = np.zeros(parameters[f"b{l}"].shape)

            S[f"sdW{l}"] = np.zeros(parameters[f"W{l}"].shape)
            S[f"sdb{l}"] = np.zeros(parameters[f"b{l}"].shape)

        return (V, S)

    def relu(self, Z):

        A = np.maximum(0, Z)
        activation_cache = Z

        return (A, activation_cache)

    def backward_relu(self, dA, Z):

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

    def softmax(self, Z):

        A = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 1e-15)

        activation_cache = Z

        return (A, activation_cache)

    def backward_softmax(self, dAL, Z):

        A = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 1e-15)
        dZ = A * (1 - A) * dAL
        return dZ

    def forward_propagation(self, X, parameters):

        A = X
        L = len(parameters) // 2
        caches = []

        for l in range(1, L):
            W = parameters[f"W{l}"]
            b = parameters[f"b{l}"]
            A_prev = A
            linear_cache = (A_prev, W, b)

            Z = np.dot(W, A_prev) + b
            A, activation_cache = self.relu(Z=Z)

            caches.append((linear_cache, activation_cache))

        W = parameters[f"W{L}"]
        b = parameters[f"b{L}"]
        A_prev = A
        linear_cache = (A_prev, W, b)

        Z = np.dot(W, A_prev) + b
        A, activation_cache = self.softmax(Z=Z)

        caches.append((linear_cache, activation_cache))

        return A, caches

    def compute_cost(self, parameters, A, Y, lambda_, dataset):

        L = len(parameters) // 2
        m = Y.shape[1]
        if dataset == "train":

            A = np.clip(A, 1e-15, 1 - 1e-15)
            loss = - np.sum(Y * np.log(A))
            L2_loss = (lambda_ / 2*m) * sum([np.sum(np.square(parameters[f"W{l}"])) for l in range(1, L + 1)])
            cost = np.squeeze(loss) + np.squeeze(L2_loss)
        else:
            A = np.clip(A, 1e-15, 1 - 1e-15)
            loss = - np.sum(Y * np.log(A))
            cost = loss

        return cost

    def backward_propagation(self, AL, Y, caches, lambda_):

        Y = Y.reshape(AL.shape)
        L = len(caches)
        gradients = {}

        dAL = - (np.divide(Y, AL + 1e-15) - np.divide(1 - Y, 1 - AL + 1e-15))

        (linear_cache, activation_cache) = caches[-1]
        (A_prev, W, b) = linear_cache
        m = A_prev.shape[1]

        dZ = self.backward_softmax(dAL=dAL, Z=activation_cache)

        gradients[f"dW{L}"] = 1. / m * np.dot(dZ, A_prev.T) + (lambda_ / m) * W
        gradients[f"db{L}"] = np.sum(dZ, axis=1, keepdims=True) / m
        gradients[f"dA{L-1}"] = np.dot(W.T, dZ)

        for l in reversed(range(L - 1)):
            (linear_cache, activation_cache) = caches[l]
            (A_prev, W, b) = linear_cache
            m = A_prev.shape[1]

            dZ = self.backward_relu(dA=gradients[f"dA{l + 1}"], Z=activation_cache)

            gradients[f"dW{l+1}"] = 1. / m * np.dot(dZ, A_prev.T) + (lambda_ / m) * W
            gradients[f"db{l+1}"] = np.sum(dZ, axis=1, keepdims=True) / m
            gradients[f"dA{l}"] = np.dot(W.T, dZ)

        return gradients

    def compute_momentum(self, parameters, gradients, V, beta, bias_correction, t):

        L = len(parameters) // 2

        for l in range(1, L + 1):

            V[f"vdW{l}"] = (beta * V[f"vdW{l}"]) + ((1 - beta) * gradients[f"dW{l}"])
            V[f"vdb{l}"] = (beta * V[f"vdb{l}"]) + ((1 - beta) * gradients[f"db{l}"])

            # bias correction terms
            if bias_correction:
                V[f"vdW{l}"] = V[f"vdW{l}"] / (1 - np.power(beta, t))
                V[f"vdb{l}"] = V[f"vdb{l}"] / (1 - np.power(beta, t))

        return V

    def compute_squared_gradients(self, parameters, gradients, S, beta, bias_correction, t):

        L = len(parameters) // 2

        for l in range(1, L + 1):

            S[f"sdW{l}"] = (beta * S[f"sdW{l}"]) + ((1 - beta) * np.power(gradients[f"dW{l}"], 2))
            S[f"sdb{l}"] = (beta * S[f"sdb{l}"]) + ((1 - beta) * np.power(gradients[f"db{l}"], 2))

            # bias correction terms
            if bias_correction:
                S[f"sdW{l}"] = S[f"sdW{l}"] / (1 - np.power(beta, t))
                S[f"sdb{l}"] = S[f"sdb{l}"] / (1 - np.power(beta, t))

        return S

    def update_parameters(self, parameters, V, S, learning_rate, epsilon):

        L = len(parameters) // 2

        for l in range(1, L + 1):
            parameters[f"W{l}"] -= learning_rate * (V[f"vdW{l}"] / np.sqrt(S[f"sdW{l}"] + epsilon))
            parameters[f"b{l}"] -= learning_rate * (V[f"vdb{l}"] / np.sqrt(S[f"sdb{l}"] + epsilon))

        return parameters

    def update_learning_rate(self, init_learning_rate, epoch_num, decay_rate, time_interval):

        learning_rate = (1 * init_learning_rate) / (1 + decay_rate * (np.floor(epoch_num / time_interval)))

        return learning_rate

    def predict(self, X, parameters):

        A, _ = self.forward_propagation(X=X, parameters=parameters)

        return A

    def accuracy(self, A, Y):

        assert (A.shape == Y.shape)

        predicted_labels = np.argmax(A, axis=0)
        true_labels = np.argmax(Y, axis=0)

        accuracy = np.mean(predicted_labels == true_labels)

        return accuracy

    def random_mini_batches(self, X, Y, mini_batch_size, seed):

        np.random.seed(seed)
        m = X.shape[1]
        n = Y.shape[0]
        mini_batches = []

        permutation = np.random.permutation(m)
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((n, m))

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

    def train(self, X_train, Y_train, X_test, Y_test, num_hidden_units, num_epochs=1000, learning_rate=1e-3, lambda_=0.0, beta1=0.9, beta2=0.999):

        if self.print_cost:
            print("___" * 45)
            print()
            print(f"Multiclass Classification FNN with {len(num_hidden_units) + 1} layers; Number of Epochs: {num_epochs}; Learning Rate: {learning_rate}; Lambda: {lambda_}, Minibatch Size: {self.minibatch_size}")
            print()
            print("___" * 45)

        train_time = 0
        self.learning_rates.append(learning_rate)

        Y_train = self.onehot_encoding(Y=Y_train)
        Y_test = self.onehot_encoding(Y=Y_test)

        num_layer_units = self.layer_sizes(
            X=X_train,
            Y=Y_train,
            num_hidden_units=num_hidden_units
        )
        parameters = self.initialize_parameters(
            num_layer_units=num_layer_units
        )
        (V, S) = self.initialize_adam_parameters(
            parameters=parameters
        )

        init_learning_rate = learning_rate
        epoch_num = 0
        m = X_train.shape[1]
        t = 0
        for i in range(num_epochs):

            self.parameters_[f"epoch {i}"] = parameters
            tic = time.time()
            self.seed = self.seed + 1

            test_mini_batches = self.random_mini_batches(
                X=X_test,
                Y=Y_test,
                mini_batch_size=self.minibatch_size,
                seed=self.seed
            )
            test_cost_total = 0.0
            for minibatch in test_mini_batches:
                (X_minibatch, Y_minibatch) = minibatch
                A_minibatch, minibatch_caches = self.forward_propagation(
                    X=X_minibatch,
                    parameters=parameters
                )

                test_cost_total += self.compute_cost(
                    parameters=parameters,
                    A=A_minibatch,
                    Y=Y_minibatch,
                    lambda_=lambda_,
                    dataset="test"
                )

            mini_batches = self.random_mini_batches(
                X=X_train,
                Y=Y_train,
                mini_batch_size=self.minibatch_size,
                seed=self.seed
            )
            cost_total = 0

            for minibatch in mini_batches:
                (X_minibatch, Y_minibatch) = minibatch
                A_minibatch, minibatch_caches = self.forward_propagation(
                    X=X_minibatch,
                    parameters=parameters
                )
                cost_total += self.compute_cost(
                    parameters=parameters,
                    A=A_minibatch,
                    Y=Y_minibatch,
                    lambda_=lambda_,
                    dataset="train"
                )
                gradients = self.backward_propagation(
                    AL=A_minibatch,
                    Y=Y_minibatch,
                    caches=minibatch_caches,
                    lambda_=lambda_,
                )
                t += 1
                V = self.compute_momentum(
                    parameters=parameters,
                    gradients=gradients,
                    V=V,
                    beta=beta1,
                    bias_correction=self.bias_correction,
                    t=t
                )
                S = self.compute_squared_gradients(
                    parameters=parameters,
                    gradients=gradients,
                    S=S,
                    beta=beta2,
                    bias_correction=self.bias_correction,
                    t=t
                )
                parameters = self.update_parameters(
                    parameters=parameters,
                    V=V,
                    S=S,
                    learning_rate=learning_rate,
                    epsilon=self.epsilon
                )

            toc = time.time()
            train_time += toc - tic
            self.costs.append(cost_total / m)
            epoch_num += 1
            learning_rate = self.update_learning_rate(
                init_learning_rate=init_learning_rate,
                epoch_num=epoch_num,
                decay_rate=self.decay_rate,
                time_interval=self.time_interval
            )
            self.learning_rates.append(learning_rate)
            if self.print_cost:

                if i % self.num_print_cost == 0 or i == num_epochs - 1:
                    self.train_cost.append(cost_total / m)
                    self.test_cost.append(test_cost_total / X_test.shape[1])
                    print(f"Epoch: {i}; Cost: {cost_total / m}")
                    train_accuracy = self.accuracy(self.predict(X=X_train, parameters=parameters), Y=Y_train)
                    test_accuracy = self.accuracy(self.predict(X=X_test, parameters=parameters), Y=Y_test)
                    self.train_accuracy.append(train_accuracy)
                    self.test_accuracy.append(test_accuracy)
                    print(f"Train Accuracy: {train_accuracy}")
                    print(f"Test Accuracy: {test_accuracy}")
                    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")

        self.time_ = train_time
        self.parameters = parameters

        if self.visualize_cost:

            plt.plot(self.costs, label="Train Cost (all epochs)")
            plt.plot(self.train_cost, label="Train Cost")
            plt.plot(self.test_cost, label="Test Cost")
            plt.title("Epoch vs. Cost")
            plt.xlabel("# Epoch")
            plt.ylabel("Cost")
            plt.legend()
            plt.show()



train_data = h5py.File("/home/samani/Documents/projects/deep-learning/data/train_signs.h5", "r")
test_data = h5py.File("/home/samani/Documents/projects/deep-learning/data/test_signs.h5", "r")
train_x = np.array(train_data["train_set_x"][:])
train_y = np.array(train_data["train_set_y"][:])
test_x = np.array(test_data["test_set_x"][:])
test_y = np.array(test_data["test_set_y"][:])
X_train = train_x.reshape(train_x.shape[0], -1).T
Y_train = train_y
X_test = test_x.reshape(test_x.shape[0], -1).T
Y_test = test_y

X_train = X_train / 255.0
X_test = X_test / 255.0


model = MulticlassClassification(
    print_cost=True,
    visualize_cost=True,
    num_print_cost=50,
    epsilon=1e-8,
    decay_rate=0.3,
    time_interval=100,
    minibatch_size=64,
    bias_correction=False,
    seed=1
)

model.train(
    X_train=X_train,
    Y_train=Y_train,
    X_test=X_test,
    Y_test=Y_test,
    num_hidden_units=[25, 15, 10],
    num_epochs=600,
    learning_rate=1e-4,
    lambda_=0.7,
    beta1=0.9,
    beta2=0.9
)


"""
__________________________________________________________________________________________________________________________

Multiclass Classification FNN with 4 layers; Number of Epochs: 600; Learning Rate: 0.0001; Lambda: 0.7, Minibatch Size: 64

__________________________________________________________________________________________________________________________
Epoch: 0; Cost: 43.49584947455967
Train Accuracy: 0.16018518518518518
Test Accuracy: 0.15
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 50; Cost: 23.522930339551454
Train Accuracy: 0.462037037037037
Test Accuracy: 0.4583333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 100; Cost: 19.304561435203173
Train Accuracy: 0.6111111111111112
Test Accuracy: 0.5333333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 150; Cost: 17.139987785530895
Train Accuracy: 0.6555555555555556
Test Accuracy: 0.5333333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 200; Cost: 15.760150842933001
Train Accuracy: 0.7083333333333334
Test Accuracy: 0.6333333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 250; Cost: 15.095574284635104
Train Accuracy: 0.75
Test Accuracy: 0.6666666666666666
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 300; Cost: 14.735473653755445
Train Accuracy: 0.7916666666666666
Test Accuracy: 0.675
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 350; Cost: 14.606144408832625
Train Accuracy: 0.7935185185185185
Test Accuracy: 0.6833333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 400; Cost: 14.58592055991462
Train Accuracy: 0.8333333333333334
Test Accuracy: 0.75
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 450; Cost: 14.60890972389157
Train Accuracy: 0.862037037037037
Test Accuracy: 0.7583333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 500; Cost: 14.688931125885825
Train Accuracy: 0.8638888888888889
Test Accuracy: 0.7583333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 550; Cost: 14.788202562407436
Train Accuracy: 0.8768518518518519
Test Accuracy: 0.7583333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 599; Cost: 14.910536346338722
Train Accuracy: 0.8907407407407407
Test Accuracy: 0.7833333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
"""

model = MulticlassClassification(
    print_cost=True,
    visualize_cost=True,
    num_print_cost=50,
    epsilon=1e-8,
    decay_rate=0.3,
    time_interval=50,
    minibatch_size=64,
    bias_correction=False,
    seed=1
)

model.train(
    X_train=X_train,
    Y_train=Y_train,
    X_test=X_test,
    Y_test=Y_test,
    num_hidden_units=[25, 15, 10],
    num_epochs=400,
    learning_rate=1e-3,
    lambda_=0.7,
    beta1=0.9,
    beta2=0.9
)

"""
__________________________________________________________________________________________________________________________

Multiclass Classification FNN with 4 layers; Number of Epochs: 400; Learning Rate: 0.001; Lambda: 0.7, Minibatch Size: 64

__________________________________________________________________________________________________________________________
Epoch: 0; Cost: 37.41047531348272
Train Accuracy: 0.16666666666666666
Test Accuracy: 0.16666666666666666
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 50; Cost: 10.333464585388024
Train Accuracy: 0.5074074074074074
Test Accuracy: 0.5333333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 100; Cost: 12.993287242225811
Train Accuracy: 0.7611111111111111
Test Accuracy: 0.6833333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 150; Cost: 14.496336536685622
Train Accuracy: 0.8351851851851851
Test Accuracy: 0.7083333333333334
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 200; Cost: 15.580734719363575
Train Accuracy: 0.9277777777777778
Test Accuracy: 0.8333333333333334
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 250; Cost: 16.246838142494916
Train Accuracy: 0.9231481481481482
Test Accuracy: 0.825
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 300; Cost: 16.572322002436067
Train Accuracy: 0.9796296296296296
Test Accuracy: 0.8833333333333333
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 350; Cost: 16.71074785434538
Train Accuracy: 0.9879629629629629
Test Accuracy: 0.8666666666666667
- - - - - - - - - - - - - - - - - - - - - - - - - - -
Epoch: 399; Cost: 16.713609326833957
Train Accuracy: 0.9712962962962963
Test Accuracy: 0.8666666666666667
- - - - - - - - - - - - - - - - - - - - - - - - - - -
"""