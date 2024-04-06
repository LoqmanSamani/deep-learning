import numpy as np
import matplotlib.pyplot as plt
import copy
import h5py
import time
import scipy.io

""" Deep Feed Forward Neural Network (FFNN) with L2 Regularization Technique for Boolean Classification Tasks"""


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

    def forward_propagation(self, X, parameters, hidden_activation, output_activation):

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the network

        for l in range(1, L):

            A_prev = A
            W = parameters[f"W{l}"]
            b = parameters[f"b{l}"]

            if hidden_activation == "relu":

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

    def compute_cost(self, A, Y, parameters, lambda_, sigma, dataset):

        m = Y.shape[1]
        L = len(parameters) // 2

        if dataset == "train" and lambda_ != 0:

            L2_cost = (lambda_ / (2 * m)) * sum([np.sum(np.square(parameters[f"W{l}"])) for l in range(1, L + 1)])
            cross_entropy_cost = - np.sum(np.multiply(Y, np.log(A + sigma)) + np.multiply(1 - Y, np.log(1 - A + sigma))) / m
            cost = np.squeeze(cross_entropy_cost) + np.squeeze(L2_cost)

        else:
            cost = - np.sum(np.multiply(Y, np.log(A + sigma)) + np.multiply(1 - Y, np.log(1 - A + sigma))) / m
            cost = np.squeeze(cost)

        return cost

    def backward_propagation(self, A, Y, caches, hidden_activation, output_activation, lambda_, sigma, dataset):

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
            if dataset == "train" and lambda_ != 0:

                dW = 1. / m * np.dot(dZ, A_prev.T) + (lambda_ / m) * W
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

            if hidden_activation == "relu":

                current_cache = caches[l]
                (linear_cache, activation_cache) = current_cache
                (A_prev, W, b) = linear_cache
                dZ = self.derivative_relu(dA=grads[f"dA{l+1}"], cache=activation_cache)

                m = A_prev.shape[1]
                if dataset == "train" and lambda_ != 0:

                    dW = 1. / m * np.dot(dZ, A_prev.T) + (lambda_ / m) * W
                    db = np.sum(dZ, axis=1, keepdims=True) / m
                    dA_prev = np.dot(W.T, dZ)

                else:

                    dW = 1. / m * np.dot(dZ, A_prev.T)
                    db = np.sum(dZ, axis=1, keepdims=True) / m
                    dA_prev = np.dot(W.T, dZ)

                grads[f"dW{l+1}"] = dW
                grads[f"db{l+1}"] = db
                grads[f"dA{l}"] = dA_prev

            else:
                raise ValueError("hidden_activation must be 'relu'!")

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

    def train(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, parameters, lambda_,
              hidden_activation, output_activation, num_epochs, learning_rate, print_cost,  epsilon, sigma):

        print()
        for i in range(1, num_epochs + 1):
            tic = time.time()

            A, caches = self.forward_propagation(
                X=X_train,
                parameters=parameters,
                hidden_activation=hidden_activation,
                output_activation=output_activation
            )
            cost = self.compute_cost(
                A=A,
                Y=Y_train,
                parameters=parameters,
                lambda_=lambda_,
                sigma=sigma,
                dataset="train"
            )
            self.train_cost.append(cost)

            if X_validation is not None:

                A_valid, _ = self.forward_propagation(
                    X=X_validation,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation
                )

                cost_v = self.compute_cost(
                    A=A_valid,
                    Y=Y_validation,
                    parameters=parameters,
                    lambda_=lambda_,
                    sigma=sigma,
                    dataset="validation"
                )
                self.validation_cost.append(cost_v)

            if X_test is not None:

                A_test, _ = self.forward_propagation(
                    X=X_test,
                    parameters=parameters,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation
                )
                cost_t = self.compute_cost(
                    A=A_test,
                    Y=Y_test,
                    parameters=parameters,
                    lambda_=lambda_,
                    sigma=sigma,
                    dataset="test"
                )
                self.test_cost.append(cost_t)

            grads = self.backward_propagation(
                A=A,
                Y=Y_train,
                caches=caches,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                lambda_=lambda_,
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

    def model(self, X_train, Y_train, num_hidden_units, X_validation=None, Y_validation=None, X_test=None, Y_test=None, lambda_=0.0, num_epochs=10000, learning_rate=1e-4,):

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
            print(f"NN_Model with {len(num_hidden_units) + 1} layers ({num_layer_units}); Number of Epochs: {num_epochs}; Learning Rate: {learning_rate}; Lambda: {lambda_}")
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
                lambda_=lambda_,
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

            if X_validation is not None:
                valid_predicted = self.predict(
                    X=X_validation,
                    parameters=self.parameters,
                    hidden_activation=self.hidden_activation,
                    output_activation=self.output_activation
                )
                valid_accuracy = self.accuracy(predicted=valid_predicted, Y=Y_validation)
                self.validation_predict = valid_predicted
                self.validation_accuracy = valid_accuracy

            if X_test is not None:
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
    lambda_=0.0,
    num_epochs=30000,
    learning_rate=1e-2
)


"""
____________________________________________________________________________________________________

NN_Model with 4 layers ([2, 10, 6, 4, 1]); Number of Epochs: 30000; Learning Rate: 0.01; Lambda: 0.0

____________________________________________________________________________________________________

Epoch 1; Cost: 0.7207915784841776; Duration: 0.00152 seconds
Epoch 1000; Cost: 0.41257319705177653; Duration: 0.56874 seconds
Epoch 2000; Cost: 0.24609675269907194; Duration: 0.4934 seconds
Epoch 3000; Cost: 0.2314591168490022; Duration: 0.50812 seconds
Epoch 4000; Cost: 0.22780725468130666; Duration: 0.49127 seconds
Epoch 5000; Cost: 0.22463297820975894; Duration: 0.49337 seconds
Epoch 6000; Cost: 0.2222533872391165; Duration: 0.49368 seconds
Epoch 7000; Cost: 0.22035086235218432; Duration: 0.49199 seconds
Epoch 8000; Cost: 0.2185880101696446; Duration: 0.49183 seconds
Epoch 9000; Cost: 0.21701602966577505; Duration: 0.48952 seconds
Epoch 10000; Cost: 0.21543104648822015; Duration: 0.49055 seconds
Epoch 11000; Cost: 0.21354184692057662; Duration: 0.49213 seconds
Epoch 12000; Cost: 0.21187591258101196; Duration: 0.50252 seconds
Epoch 13000; Cost: 0.21020260391940152; Duration: 0.49216 seconds
Epoch 14000; Cost: 0.20830400628196985; Duration: 0.49319 seconds
Epoch 15000; Cost: 0.20604511204104628; Duration: 0.49518 seconds
Epoch 16000; Cost: 0.2028261325282489; Duration: 0.49066 seconds
Epoch 17000; Cost: 0.19842843437902247; Duration: 0.49322 seconds
Epoch 18000; Cost: 0.19615418897580986; Duration: 0.50399 seconds
Epoch 19000; Cost: 0.1943732736360888; Duration: 0.49022 seconds
Epoch 20000; Cost: 0.19169905598086548; Duration: 0.49487 seconds
Epoch 21000; Cost: 0.1902006195112381; Duration: 0.49499 seconds
Epoch 22000; Cost: 0.18896178757138676; Duration: 0.4937 seconds
Epoch 23000; Cost: 0.18787298346911446; Duration: 0.49232 seconds
Epoch 24000; Cost: 0.1868782803175432; Duration: 0.49404 seconds
Epoch 25000; Cost: 0.186021684888526; Duration: 0.4958 seconds
Epoch 26000; Cost: 0.18449610187807672; Duration: 0.49144 seconds
Epoch 27000; Cost: 0.18345464869337966; Duration: 0.49831 seconds
Epoch 28000; Cost: 0.18252458496457183; Duration: 0.49402 seconds
Epoch 29000; Cost: 0.1816612893733618; Duration: 0.49041 seconds
Epoch 30000; Cost: 0.18085265073538945; Duration: 0.48848 seconds

________________________________________

Train Accuracy: 0.9383886255924171
Test Accuracy: 0.9499999999999998

________________________________________

"""


obj.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[10, 6, 4],
    X_test=X_test,
    Y_test=Y_test,
    lambda_=0.86,
    num_epochs=30000,
    learning_rate=1e-2
)


"""
______________________________________________________________________________________________________

NN_Model with 4 layers ([2, 10, 6, 4, 1]); Number of Epochs: 30000; Learning Rate: 0.01; Lambda: 0.86

______________________________________________________________________________________________________

Epoch 1; Cost: 0.7477530168127471; Duration: 0.00207 seconds
Epoch 1000; Cost: 0.3439089580399352; Duration: 0.61145 seconds
Epoch 2000; Cost: 0.32624865184811325; Duration: 0.57012 seconds
Epoch 3000; Cost: 0.3200742564526751; Duration: 0.71276 seconds
Epoch 4000; Cost: 0.31487179189886244; Duration: 0.58769 seconds
Epoch 5000; Cost: 0.31036123341046196; Duration: 0.55983 seconds
Epoch 6000; Cost: 0.30635435317834925; Duration: 0.55852 seconds
Epoch 7000; Cost: 0.302717117060929; Duration: 0.55993 seconds
Epoch 8000; Cost: 0.2994579721635692; Duration: 0.56013 seconds
Epoch 9000; Cost: 0.29650084884324274; Duration: 0.5606 seconds
Epoch 10000; Cost: 0.2938285454428393; Duration: 0.55414 seconds
Epoch 11000; Cost: 0.29141356898963444; Duration: 0.55802 seconds
Epoch 12000; Cost: 0.28922580306191614; Duration: 0.56046 seconds
Epoch 13000; Cost: 0.287237317243489; Duration: 0.55923 seconds
Epoch 14000; Cost: 0.2854309240660733; Duration: 0.55896 seconds
Epoch 15000; Cost: 0.2837855953705153; Duration: 0.55757 seconds
Epoch 16000; Cost: 0.28228471742615246; Duration: 0.55982 seconds
Epoch 17000; Cost: 0.28091511687940385; Duration: 0.55996 seconds
Epoch 18000; Cost: 0.2796627895470074; Duration: 0.55776 seconds
Epoch 19000; Cost: 0.27852104859975435; Duration: 0.55874 seconds
Epoch 20000; Cost: 0.277489424806775; Duration: 0.55701 seconds
Epoch 21000; Cost: 0.27654059761051203; Duration: 0.56682 seconds
Epoch 22000; Cost: 0.2756763011256284; Duration: 0.56419 seconds
Epoch 23000; Cost: 0.2748798762446282; Duration: 0.5592 seconds
Epoch 24000; Cost: 0.27413257708680955; Duration: 0.56642 seconds
Epoch 25000; Cost: 0.2734074569793426; Duration: 0.56568 seconds
Epoch 26000; Cost: 0.27274727610991856; Duration: 0.55841 seconds
Epoch 27000; Cost: 0.2721412247412238; Duration: 0.56164 seconds
Epoch 28000; Cost: 0.2716000343851249; Duration: 0.56334 seconds
Epoch 29000; Cost: 0.271102861601829; Duration: 0.56118 seconds
Epoch 30000; Cost: 0.27065129166696705; Duration: 0.5575 seconds

________________________________________

Train Accuracy: 0.9289099526066351
Test Accuracy: 0.9249999999999998

________________________________________
"""


obj.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[10, 6, 4],
    X_test=X_test,
    Y_test=Y_test,
    lambda_=0.0,
    num_epochs=30000,
    learning_rate=0.2
)


"""
___________________________________________________________________________________________________

NN_Model with 4 layers ([2, 10, 6, 4, 1]); Number of Epochs: 30000; Learning Rate: 0.2; Lambda: 0.0

___________________________________________________________________________________________________

Epoch 1; Cost: 0.6855105755965918; Duration: 0.00151 seconds
Epoch 1000; Cost: 0.1940940223764738; Duration: 0.57846 seconds
Epoch 2000; Cost: 0.18320609465955465; Duration: 0.53474 seconds
Epoch 3000; Cost: 0.17326959982107562; Duration: 0.56809 seconds
Epoch 4000; Cost: 0.166206919271973; Duration: 0.55003 seconds
Epoch 5000; Cost: 0.1654777180875574; Duration: 0.50704 seconds
Epoch 6000; Cost: 0.16481253062457932; Duration: 0.51086 seconds
Epoch 7000; Cost: 0.16021140246965063; Duration: 0.49902 seconds
Epoch 8000; Cost: 0.15042463531050973; Duration: 0.50613 seconds
Epoch 9000; Cost: 0.14706009988810817; Duration: 0.49685 seconds
Epoch 10000; Cost: 0.14018076794802098; Duration: 0.50275 seconds
Epoch 11000; Cost: 0.15650868353283318; Duration: 0.50056 seconds
Epoch 12000; Cost: 0.122207077397888; Duration: 0.50021 seconds
Epoch 13000; Cost: 0.1120660690952354; Duration: 0.50221 seconds
Epoch 14000; Cost: 0.2783664385458048; Duration: 0.50021 seconds
Epoch 15000; Cost: 0.11561726187430665; Duration: 0.49954 seconds
Epoch 16000; Cost: 0.12278610787785417; Duration: 0.50158 seconds
Epoch 17000; Cost: 0.1270051966659158; Duration: 0.51379 seconds
Epoch 18000; Cost: 0.2301996533540642; Duration: 0.50135 seconds
Epoch 19000; Cost: 0.10907752958129678; Duration: 0.49762 seconds
Epoch 20000; Cost: 0.11446006661160166; Duration: 0.50323 seconds
Epoch 21000; Cost: 0.2062746150744032; Duration: 0.50675 seconds
Epoch 22000; Cost: 0.08940267601848667; Duration: 0.50622 seconds
Epoch 23000; Cost: 0.0800805021709509; Duration: 0.50215 seconds
Epoch 24000; Cost: 0.08084273592994283; Duration: 0.50203 seconds
Epoch 25000; Cost: 0.07992220209604523; Duration: 0.50536 seconds
Epoch 26000; Cost: 0.09371500667158997; Duration: 0.50362 seconds
Epoch 27000; Cost: 0.12532818701128462; Duration: 0.51321 seconds
Epoch 28000; Cost: 0.06650640219729825; Duration: 0.51081 seconds
Epoch 29000; Cost: 0.06581013959447919; Duration: 0.50631 seconds
Epoch 30000; Cost: 0.14737095023844612; Duration: 0.50458 seconds

________________________________________

Train Accuracy: 0.9478672985781991
Test Accuracy: 0.8949999999999999

________________________________________
"""

obj.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[10, 6, 4],
    X_test=X_test,
    Y_test=Y_test,
    lambda_=0.70,
    num_epochs=30000,
    learning_rate=0.2
)

"""
_____________________________________________________________________________________________________

NN_Model with 4 layers ([2, 10, 6, 4, 1]); Number of Epochs: 30000; Learning Rate: 0.2; Lambda: 0.7

______________________________________________________________________________________________________

Epoch 1; Cost: 0.7143; Duration: 0.00161 seconds
Epoch 1000; Cost: 0.26713; Duration: 0.60225 seconds
Epoch 2000; Cost: 0.2603; Duration: 0.56451 seconds
Epoch 3000; Cost: 0.25938; Duration: 0.64779 seconds
Epoch 4000; Cost: 0.25674; Duration: 0.61185 seconds
Epoch 5000; Cost: 0.26641; Duration: 0.70345 seconds
Epoch 6000; Cost: 0.2548; Duration: 0.56897 seconds
Epoch 7000; Cost: 0.25713; Duration: 0.55826 seconds
Epoch 8000; Cost: 0.26037; Duration: 0.55874 seconds
Epoch 9000; Cost: 0.26273; Duration: 0.55671 seconds
Epoch 10000; Cost: 0.26404; Duration: 0.55718 seconds
Epoch 11000; Cost: 0.25202; Duration: 0.55438 seconds
Epoch 12000; Cost: 0.25208; Duration: 0.55878 seconds
Epoch 13000; Cost: 0.25174; Duration: 0.58445 seconds
Epoch 14000; Cost: 0.25198; Duration: 0.58542 seconds
Epoch 15000; Cost: 0.25211; Duration: 0.63254 seconds
Epoch 16000; Cost: 0.28383; Duration: 0.5748 seconds
Epoch 17000; Cost: 0.25258; Duration: 0.56523 seconds
Epoch 18000; Cost: 0.25411; Duration: 0.56117 seconds
Epoch 19000; Cost: 0.26972; Duration: 0.56237 seconds
Epoch 20000; Cost: 0.25211; Duration: 0.56561 seconds
Epoch 21000; Cost: 0.25068; Duration: 0.59101 seconds
Epoch 22000; Cost: 0.2711; Duration: 0.57138 seconds
Epoch 23000; Cost: 0.25942; Duration: 0.5629 seconds
Epoch 24000; Cost: 0.25764; Duration: 0.56663 seconds
Epoch 25000; Cost: 0.25269; Duration: 0.56808 seconds
Epoch 26000; Cost: 0.25468; Duration: 0.56932 seconds
Epoch 27000; Cost: 0.25184; Duration: 0.56423 seconds
Epoch 28000; Cost: 0.2551; Duration: 0.55858 seconds
Epoch 29000; Cost: 0.25187; Duration: 0.56119 seconds
Epoch 30000; Cost: 0.25236; Duration: 0.56519 seconds

________________________________________

Train Accuracy: 0.933649289099526
Test Accuracy: 0.9399999999999998

________________________________________

"""