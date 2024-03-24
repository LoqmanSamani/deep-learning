import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import time



""" A neural network with one hidden layer for classification tasks """


class NNClassification(object):

    def __init__(self, print_cost=False, hidden_activation="relu", epsilon=100, theta=1e-2, sigma=1e-12, seed=42):

        self.print_cost = print_cost
        self.hidden_activation = hidden_activation
        self.theta = theta
        self.epsilon = epsilon
        self.sigma = sigma
        self.seed = seed


        self.parameters = {}
        self.costs = []
        self.durations = []


    def layer_sizes(self, X, Y, num_hidden_units):

        n_x = X.shape[0]
        n_h = num_hidden_units
        n_y = Y.shape[0]

        return (n_x, n_h, n_y)


    def initialize_parameters(self, n_x, n_h, n_y, theta, seed):

        np.random.seed(seed)

        W1 = np.random.randn(n_h, n_x) * theta
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * theta
        b2 = np.zeros((n_y, 1))

        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }
        return parameters


    def sigmoid(self, Z2, sigma):

        A2 = 1 / (1 + np.exp(-Z2 + sigma))

        return A2


    def forward_propagation(self, X, parameters, hidden_activation, sigma):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        if hidden_activation == "relu":
            A1 = np.maximum(0, Z1)
        else:
            A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2

        A2 = self.sigmoid(Z2, sigma)

        cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }

        return A2, cache


    def compute_cost(self, Y, A2, sigma):

        m = Y.shape[1]

        log_loss = np.multiply(np.log(A2 + sigma), Y) + np.multiply(np.log(1 - A2 + sigma), (1 - Y + sigma))
        cost = - np.sum(log_loss) / m

        cost = float(np.squeeze(cost))  # makes sure cost is the dimension you expect.

        return cost



    def backward_propagation(self, X, Y, parameters, cache):

        m = X.shape[1]

        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return grads


    def update_parameters(self, parameters, grads, alpha):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2

        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        return parameters



    def train(self, X, Y, parameters, hidden_activation, sigma, epochs, alpha, print_cost, epsilon):

        for i in range(epochs):

            tic = time.time()

            A2, cache = self.forward_propagation(
                X=X,
                parameters=parameters,
                hidden_activation=hidden_activation,
                sigma=sigma
            )

            cost = self.compute_cost(
                Y=Y,
                A2=A2,
                sigma=sigma
            )

            self.costs.append(cost)

            grads = self.backward_propagation(
                X=X,
                Y=Y,
                parameters=parameters,
                cache=cache
            )

            parameters = self.update_parameters(
                parameters=parameters,
                grads=grads,
                alpha=alpha
            )

            toc = time.time()

            duration = toc - tic
            self.durations.append(duration)

            if print_cost and i % epsilon == 0:
                message = f"Epoch {i}; Cost: {cost}; Duration: {round(sum(self.durations[i-epsilon+1:i+1]), 3)} S"
                print(message)

        return parameters



    def predict(self, X, parameters):

        A2, cache = self.forward_propagation(
            X=X,
            parameters=parameters,
            hidden_activation=self.hidden_activation,
            sigma=self.sigma
        )

        predictions = np.where(A2 > 0.5, 1, 0)

        return predictions



    def accuracy(self, parameters, X_train, Y_train, X_test, Y_test):

        train_predict = self.predict(X=X_train, parameters=parameters)
        test_predict = self.predict(X=X_test, parameters=parameters)

        train_set_accuracy = np.mean(train_predict == Y_train) * 100
        test_set_accuracy = np.mean(test_predict == Y_test) * 100

        return train_set_accuracy, test_set_accuracy



    def nn_model(self, X_train, Y_train, X_test=None, Y_test=None,  accuracy=True, num_hidden_units=5, epochs=1000, alpha=1e-3):

        sizes = self.layer_sizes(
            X=X_train,
            Y=Y_train,
            num_hidden_units=num_hidden_units
        )

        parameters = self.initialize_parameters(
            n_x=sizes[0],
            n_h=sizes[1],
            n_y=sizes[2],
            theta=self.theta,
            seed=self.seed
        )

        print(f"Number of Epochs: {epochs}; Learning Rate: {alpha}; Number of Units in the Hidden Layer: {num_hidden_units}")
        print("__" * 50)

        parameters = self.train(
            X=X_train,
            Y=Y_train,
            parameters=parameters,
            hidden_activation=self.hidden_activation,
            sigma=self.sigma,
            epochs=epochs,
            alpha=alpha,
            print_cost=self.print_cost,
            epsilon=self.epsilon
        )
        self.parameters = parameters

        if accuracy:

            train_set_accuracy, test_set_accuracy = self.accuracy(
                parameters=parameters,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test
            )
            print("__" * 25)
            print(f"Train Set Accuracy: {train_set_accuracy}")
            print(f"Test Set Accuracy: {test_set_accuracy}")
            print("__" * 25)







"""Generate Data"""

N = 4000
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


dataset = "noisy_moons"


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
(2, 4000)
(1, 4000)
"""
X_train = X[:, :3500]
Y_train = Y[:, :3500]
X_test = X[:, 3500:]
Y_test = Y[:, 3500:]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
"""
(2, 3500)
(1, 3500)
(2, 500)
(1, 500)
"""

"""
model = NNClassification(
    print_cost=True,
    hidden_activation="relu",
    epsilon=1000,
    theta=1e-2,
    sigma=1e-12,
    seed=42
)

model.nn_model(
    X_train=X_train,
    Y_train=Y_train,
    X_test=X_test,
    Y_test=Y_test,
    num_hidden_units=5,
    epochs=10000,
    alpha=0.01
)
"""

"""
Number of Epochs: 10000, Learning Rate: 0.01, Number of Units in the Hidden Layer: 5
________________________________________________________________________________
Epoch 0; Cost: 0.6931612495016675, Duration: 0.0017924308776855469 Seconds
Epoch 1000; Cost: 0.69076327026119, Duration: 1.7514491081237793 Seconds
Epoch 2000; Cost: 0.6243387496209203, Duration: 2.1813323497772217 Seconds
Epoch 3000; Cost: 0.46846700259380447, Duration: 1.4071228504180908 Seconds
Epoch 4000; Cost: 0.39583926662153407, Duration: 1.6589195728302002 Seconds
Epoch 5000; Cost: 0.3730481853300045, Duration: 1.550217866897583 Seconds
Epoch 6000; Cost: 0.3643680671248189, Duration: 1.5055675506591797 Seconds
Epoch 7000; Cost: 0.3603496402635423, Duration: 1.3479433059692383 Seconds
Epoch 8000; Cost: 0.3554304871149896, Duration: 1.4740562438964844 Seconds
Epoch 9000; Cost: 0.34704488306568615, Duration: 1.5339155197143555 Seconds
________________________________________________________________________________
Train Set Accuracy: 83.17142857142858
Test Set Accuracy: 86.2
________________________________________________________________________________

"""



learning_rates = [0.1, 0.05, 0.01, 0.001]


for alpha in learning_rates:
    model = NNClassification(
        print_cost=True,
        hidden_activation="relu",
        epsilon=1000,
        theta=1e-2,
        sigma=1e-12,
        seed=42
    )

    model.nn_model(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        num_hidden_units=5,
        epochs=10000,
        alpha=alpha
    )


"""
Number of Epochs: 10000; Learning Rate: 0.1; Number of Units in the Hidden Layer: 5
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931612495016675; Duration: 0.002 S
Epoch 1000; Cost: 0.3457968129379629; Duration: 1.892 S
Epoch 2000; Cost: 0.33070455516478114; Duration: 1.556 S
/home/samani/Documents/projects/deep-learning/models/classification.py:55: RuntimeWarning: overflow encountered in exp
  A2 = 1 / (1 + np.exp(-Z2 + sigma))
/home/samani/Documents/projects/deep-learning/models/classification.py:107: RuntimeWarning: overflow encountered in multiply
  dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
/home/samani/Documents/projects/new/lib/python3.10/site-packages/numpy/core/fromnumeric.py:88: RuntimeWarning: overflow encountered in reduce
  return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
Epoch 3000; Cost: nan; Duration: 1.552 S
Epoch 4000; Cost: nan; Duration: 1.29 S
Epoch 5000; Cost: nan; Duration: 1.268 S
Epoch 6000; Cost: nan; Duration: 1.373 S
Epoch 7000; Cost: nan; Duration: 1.388 S
Epoch 8000; Cost: nan; Duration: 1.451 S
Epoch 9000; Cost: nan; Duration: 1.301 S
__________________________________________________
Train Set Accuracy: 50.25714285714285
Test Set Accuracy: 48.199999999999996
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.05; Number of Units in the Hidden Layer: 5
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931612495016675; Duration: 0.002 S
Epoch 1000; Cost: 0.37321355812148027; Duration: 2.73 S
Epoch 2000; Cost: 0.34570587504316286; Duration: 1.752 S
Epoch 3000; Cost: 0.3382634629352329; Duration: 1.551 S
Epoch 4000; Cost: 0.3305840476119107; Duration: 1.57 S
/home/samani/Documents/projects/deep-learning/models/classification.py:107: RuntimeWarning: overflow encountered in power
  dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
/home/samani/Documents/projects/deep-learning/models/classification.py:107: RuntimeWarning: invalid value encountered in multiply
  dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
Epoch 5000; Cost: nan; Duration: 1.667 S
Epoch 6000; Cost: nan; Duration: 1.358 S
Epoch 7000; Cost: nan; Duration: 1.342 S
Epoch 8000; Cost: nan; Duration: 1.278 S
Epoch 9000; Cost: nan; Duration: 1.359 S
__________________________________________________
Train Set Accuracy: 50.25714285714285
Test Set Accuracy: 48.199999999999996
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.01; Number of Units in the Hidden Layer: 5
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931612495016675; Duration: 0.001 S
Epoch 1000; Cost: 0.69076327026119; Duration: 1.725 S
Epoch 2000; Cost: 0.6243387496209203; Duration: 1.717 S
Epoch 3000; Cost: 0.46846700259380447; Duration: 1.534 S
Epoch 4000; Cost: 0.39583926662153407; Duration: 1.567 S
Epoch 5000; Cost: 0.3730481853300045; Duration: 1.517 S
Epoch 6000; Cost: 0.3643680671248189; Duration: 1.602 S
Epoch 7000; Cost: 0.3603496402635423; Duration: 2.459 S
Epoch 8000; Cost: 0.3554304871149896; Duration: 1.792 S
Epoch 9000; Cost: 0.34704488306568615; Duration: 1.555 S
__________________________________________________
Train Set Accuracy: 83.17142857142858
Test Set Accuracy: 86.2
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.001; Number of Units in the Hidden Layer: 5
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931612495016675; Duration: 0.003 S
Epoch 1000; Cost: 0.6931097731103448; Duration: 2.086 S
Epoch 2000; Cost: 0.6930548143639547; Duration: 1.661 S
Epoch 3000; Cost: 0.6929892031732576; Duration: 1.609 S
Epoch 4000; Cost: 0.69290530945145; Duration: 1.841 S
Epoch 5000; Cost: 0.6927914633098087; Duration: 1.663 S
Epoch 6000; Cost: 0.6926279448470558; Duration: 1.603 S
Epoch 7000; Cost: 0.6923882242823398; Duration: 1.825 S
Epoch 8000; Cost: 0.6920353931296673; Duration: 1.73 S
Epoch 9000; Cost: 0.6915164286246137; Duration: 1.635 S
__________________________________________________
Train Set Accuracy: 50.25714285714285
Test Set Accuracy: 48.199999999999996
__________________________________________________

"""



units = [2, 4, 6, 8, 10, 20]


for unit in units:
    model = NNClassification(
        print_cost=True,
        hidden_activation="relu",
        epsilon=1000,
        theta=1e-2,
        sigma=1e-12,
        seed=42
    )

    model.nn_model(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        num_hidden_units=unit,
        epochs=10000,
        alpha=0.01
    )


"""
Number of Epochs: 10000; Learning Rate: 0.01; Number of Units in the Hidden Layer: 2
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931466214840925; Duration: 0.001 S
Epoch 1000; Cost: 0.6928530464681645; Duration: 0.528 S
Epoch 2000; Cost: 0.678759122585126; Duration: 0.525 S
Epoch 3000; Cost: 0.524924369631232; Duration: 0.513 S
Epoch 4000; Cost: 0.4336625707404896; Duration: 0.5 S
Epoch 5000; Cost: 0.39340662521855885; Duration: 0.632 S
Epoch 6000; Cost: 0.3663211164977526; Duration: 0.506 S
Epoch 7000; Cost: 0.34806115644368724; Duration: 0.517 S
Epoch 8000; Cost: 0.3357243356279887; Duration: 0.491 S
Epoch 9000; Cost: 0.3299138559962417; Duration: 0.498 S
__________________________________________________
Train Set Accuracy: 83.77142857142857
Test Set Accuracy: 86.2
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.01; Number of Units in the Hidden Layer: 4
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931658143000241; Duration: 0.001 S
Epoch 1000; Cost: 0.6930362209259728; Duration: 1.213 S
Epoch 2000; Cost: 0.6840669073059691; Duration: 1.196 S
Epoch 3000; Cost: 0.5199049521927996; Duration: 1.302 S
Epoch 4000; Cost: 0.395552312583956; Duration: 1.227 S
Epoch 5000; Cost: 0.3538189950128329; Duration: 1.275 S
Epoch 6000; Cost: 0.33332072685190295; Duration: 1.175 S
Epoch 7000; Cost: 0.32175211615985544; Duration: 1.43 S
Epoch 8000; Cost: 0.31481952553007325; Duration: 1.549 S
Epoch 9000; Cost: 0.31055594043613655; Duration: 1.868 S
__________________________________________________
Train Set Accuracy: 84.11428571428571
Test Set Accuracy: 84.0
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.01; Number of Units in the Hidden Layer: 6
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931201008507256; Duration: 0.003 S
Epoch 1000; Cost: 0.6897424023012788; Duration: 1.672 S
Epoch 2000; Cost: 0.5897725678459691; Duration: 1.637 S
Epoch 3000; Cost: 0.4152090631589746; Duration: 2.081 S
Epoch 4000; Cost: 0.359363552492437; Duration: 1.593 S
Epoch 5000; Cost: 0.33586841339646845; Duration: 1.617 S
Epoch 6000; Cost: 0.325311916751237; Duration: 1.946 S
Epoch 7000; Cost: 0.32032642168902825; Duration: 2.065 S
Epoch 8000; Cost: 0.3184487356854078; Duration: 1.859 S
Epoch 9000; Cost: 0.3176147762485746; Duration: 1.633 S
__________________________________________________
Train Set Accuracy: 84.17142857142858
Test Set Accuracy: 87.4
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.01; Number of Units in the Hidden Layer: 8
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931821595905774; Duration: 0.002 S
Epoch 1000; Cost: 0.6905045897594703; Duration: 1.856 S
Epoch 2000; Cost: 0.5796870551354082; Duration: 1.986 S
Epoch 3000; Cost: 0.3929522784475907; Duration: 2.457 S
Epoch 4000; Cost: 0.3454932942333603; Duration: 1.968 S
Epoch 5000; Cost: 0.3274966773914637; Duration: 1.819 S
Epoch 6000; Cost: 0.3187895269982347; Duration: 1.831 S
Epoch 7000; Cost: 0.31429254045194593; Duration: 1.965 S
Epoch 8000; Cost: 0.31181815266589635; Duration: 1.75 S
Epoch 9000; Cost: 0.31037562249329675; Duration: 2.021 S
__________________________________________________
Train Set Accuracy: 84.2
Test Set Accuracy: 87.8
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.01; Number of Units in the Hidden Layer: 10
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931626554070437; Duration: 0.011 S
Epoch 1000; Cost: 0.6890937754787612; Duration: 2.164 S
Epoch 2000; Cost: 0.5727777775814068; Duration: 2.259 S
Epoch 3000; Cost: 0.41599754456930765; Duration: 2.239 S
Epoch 4000; Cost: 0.3673920057974442; Duration: 2.236 S
Epoch 5000; Cost: 0.34765395655447395; Duration: 2.478 S
Epoch 6000; Cost: 0.33793145316328566; Duration: 2.627 S
Epoch 7000; Cost: 0.3325206463754586; Duration: 2.229 S
Epoch 8000; Cost: 0.32917060642251583; Duration: 2.27 S
Epoch 9000; Cost: 0.3267195387318109; Duration: 2.25 S
__________________________________________________
Train Set Accuracy: 83.62857142857143
Test Set Accuracy: 86.8
__________________________________________________
Number of Epochs: 10000; Learning Rate: 0.01; Number of Units in the Hidden Layer: 20
____________________________________________________________________________________________________
Epoch 0; Cost: 0.6931978427034361; Duration: 0.004 S
Epoch 1000; Cost: 0.6867575081102414; Duration: 3.7 S
Epoch 2000; Cost: 0.5132849324342644; Duration: 3.943 S
Epoch 3000; Cost: 0.3595196726629867; Duration: 4.125 S
Epoch 4000; Cost: 0.32189475657643274; Duration: 4.905 S
Epoch 5000; Cost: 0.30821088187178386; Duration: 3.888 S
Epoch 6000; Cost: 0.3020469138576092; Duration: 3.95 S
Epoch 7000; Cost: 0.2990643964471384; Duration: 3.923 S
Epoch 8000; Cost: 0.29758289708799984; Duration: 4.075 S
Epoch 9000; Cost: 0.29683174011256935; Duration: 3.927 S
__________________________________________________
Train Set Accuracy: 86.0
Test Set Accuracy: 86.4
__________________________________________________

"""

