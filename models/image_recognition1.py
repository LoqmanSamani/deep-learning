import numpy as np
import copy
import h5py


""" Image Recognition with Neural Network (logistic regression) """


class ImageRecognition(object):

    def __init__(self, epochs=1000, alpha=1e-3, print_cost=True, lambda_=1e-4, epsilon=100, sigma=1e-12):

        self.epochs = epochs
        self.alpha = alpha
        self.print_cost = print_cost
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.sigma = sigma

        self.costs = []
        self.parameters = None


    def layer_size(self, X, Y):

        m = X.shape[0]
        n = Y.shape[0]

        return m, n


    def init_params(self, m, n, lambda_):

        W = np.random.randn(m, n) * lambda_
        b = np.zeros((n, 1))

        params = {
            "W": W,
            "b": b
        }

        return params



    def sigmoid(self, Z):

        A = 1 / (1 + np.exp(-Z))

        return A



    def compute_cost(self, Y, A, sigma):

        cost = - np.mean(Y * np.log(A + sigma) + ((1 - Y) * np.log(1 - A + sigma)))

        return cost



    def forward_propagation(self, W, X, b):

        Z = np.dot(W.T, X) + b

        return Z



    def propagate(self, W, b, X, Y):

        m = X.shape[1]
        Z = self.forward_propagation(W=W, X=X, b=b)
        A = self.sigmoid(Z=Z)
        cost = self.compute_cost(Y=Y, A=A, sigma=self.sigma)

        dA = A - Y
        dW = np.dot(X, dA.T) / m
        db = np.mean(dA)

        cost = np.squeeze(np.array(cost))

        grads = {"dW": dW,
                 "db": db}

        return grads, cost



    def optimize(self, W, b, X, Y, epochs, alpha, print_cost, epsilon):

        W = copy.deepcopy(W)
        b = copy.deepcopy(b)

        for i in range(epochs):

            grads, cost = self.propagate(W=W, b=b, X=X, Y=Y)

            dW = grads["dW"]
            db = grads["db"]

            W -= alpha * dW
            b -= alpha * db

            if i % epsilon == 0:
                self.costs.append(cost)

                if print_cost:
                    print(f"Cost after iteration {i}: {cost}")

        parameters = {"W": W,
                      "b": b}

        grads = {"dW": dW,
                 "db": db}

        return parameters, grads



    def predict(self, W, b, X):

        W = W.reshape(X.shape[0], 1)
        Z = self.forward_propagation(
            W=W,
            X=X,
            b=b
        )
        A = self.sigmoid(Z=Z)
        Y_prediction = np.where(A > 0.5, 1, 0)

        return Y_prediction



    def model(self, X_train, Y_train, X_test, Y_test):

        m, n = self.layer_size(
            X=X_train,
            Y=Y_train
        )
        params = self.init_params(
            m=m,
            n=n,
            lambda_=self.lambda_
        )
        params, grads = self.optimize(
            W=params["W"],
            b=params["b"],
            X=X_train,
            Y=Y_train,
            epochs=self.epochs,
            alpha=self.alpha,
            print_cost=self.print_cost,
            epsilon=self.epsilon
        )

        Y_prediction_test = self.predict(W=params["W"], b=params["b"], X=X_test)
        Y_prediction_train = self.predict(W=params["W"], b=params["b"], X=X_train)

        self.parameters = params

        if self.print_cost:
            print(f"train accuracy: {100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100}")
            print(f"test accuracy: {100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100}")






"""Load Data"""

train_dataset = h5py.File("/home/sam/projects/deep-learning/data/train_cat.h5", "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

test_dataset = h5py.File("/home/sam/projects/deep-learning/data/test_cat.h5", "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

classes = np.array(test_dataset["list_classes"][:])  # the list of classes

train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))





"""
Reshape the training and test data sets 
so that images of size (num_px, num_px, 3) 
are flattened into single vectors of shape 
(num_px ∗ num_px ∗ 3, 1)
"""

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print(train_set_x_flatten.shape)
print(test_set_x_flatten.shape)





"""
To represent color images, the red, green and blue channels (RGB) 
must be specified for each pixel, and so the pixel value is 
actually a vector of three numbers ranging from 0 to 255.

One common preprocessing step in machine learning is 
to center and standardize your dataset, meaning that you 
substract the mean of the whole numpy array from each example, 
and then divide each example by the standard deviation of the 
whole numpy array. But for picture datasets, it is simpler and 
more convenient and works almost as well to just divide every row 
of the dataset by 255 (the maximum value of a pixel channel).
"""

X_train = train_set_x_flatten / 255
Y_train = train_set_y_orig
X_test = test_set_x_flatten / 255
Y_test = test_set_y_orig
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
"""
(12288, 209)
(12288, 50)
(12288, 209)
(1, 209)
(12288, 50)
(1, 50)
"""




"""Train the model"""
image_recognition = ImageRecognition(
    epochs=2000,
    alpha=0.005,
    print_cost=True,
    lambda_=1e-4,
    epsilon=100,
    sigma=1e-12
)

image_recognition.model(
    X_train=X_train,
    Y_train=Y_train,
    X_test=X_test,
    Y_test=Y_test
)

"""
Cost after iteration 0: 0.6936515159302786
Cost after iteration 100: 0.5845168957348857
Cost after iteration 200: 0.4669515147674651
Cost after iteration 300: 0.3760072157863372
Cost after iteration 400: 0.33146494237914376
Cost after iteration 500: 0.30327534189993854
Cost after iteration 600: 0.279882289044614
Cost after iteration 700: 0.2600450999130246
Cost after iteration 800: 0.24294378449242746
Cost after iteration 900: 0.22800737274503668
Cost after iteration 1000: 0.21482265530196507
Cost after iteration 1100: 0.20308128379425303
Cost after iteration 1200: 0.19254729583368185
Cost after iteration 1300: 0.18303626529886932
Cost after iteration 1400: 0.17440142010776677
Cost after iteration 1500: 0.16652411548992663
Cost after iteration 1600: 0.15930712699488564
Cost after iteration 1700: 0.15266982349420402
Cost after iteration 1800: 0.14654462532223675
Cost after iteration 1900: 0.140874360010493
train accuracy: 99.04306220095694
test accuracy: 70.0
"""

