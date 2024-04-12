import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from tensorflow.keras.initializers import GlorotNormal, HeNormal, VarianceScaling
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow import GradientTape


class ImageRecognition:
    def __init__(self, optimizer="Adam", initializer="Xavier", print_cost=False, visualize_cost=False, num_print_cost=100, minibatch_size=64, seed=1):
        self.optimizer = optimizer
        self.initializer = initializer
        self.print_cost = print_cost
        self.visualize_cost = visualize_cost
        self.num_print_cost = num_print_cost
        self.minibatch_size = minibatch_size
        self.seed = seed

        self.parameters = {}
        self.train_cost = []
        self.validation_cost = []
        self.test_cost = []
        self.train_duration = []

    def layer_sizes(self, X, Y, num_hidden_units):

        num_input_units = X.element_spec.shape[0]
        num_output_units = Y.element_spec.shape[0]

        num_layer_units = [num_input_units] + num_hidden_units + [num_output_units]

        return num_layer_units

    def initialize_parameters(self, num_layer_units, seed):

        parameters = {}
        L = len(num_layer_units)

        initializer = GlorotNormal(seed=seed)

        if self.optimizer == "Xavier":
            initializer = GlorotNormal(seed=seed)

        elif self.optimizer == "He":
            initializer = HeNormal(seed=seed)

        elif self.optimizer == "Variance":
            initializer = VarianceScaling(seed=seed)

        for l in range(1, L):

            parameters[f"W{l}"] = tf.Variable(initializer(shape=(num_layer_units[l], num_layer_units[l - 1])))
            parameters[f"b{l}"] = tf.Variable(initializer(shape=(num_layer_units[l], 1)))

        return parameters

    def forward_propagation(self, X, parameters):

        L = len(parameters) // 2
        A = X

        for l in range(1, L):
            A_prev = A
            W = parameters[f"W{l}"]
            b = parameters[f"b{l}"]
            Z = tf.math.add(tf.linalg.matmul(W, A_prev), b)
            A = relu(Z)

        A_prev = A
        W = parameters[f"W{L}"]
        b = parameters[f"b{L}"]
        ZL = tf.math.add(tf.linalg.matmul(W, A_prev), b)

        return ZL

    def compute_cost(self, ZL, Y):

        cost = tf.reduce_sum(
            categorical_crossentropy(
                tf.transpose(Y),
                tf.transpose(ZL),
                from_logits=True
            )
        )

        return cost

    def train(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, parameters, optimizer, train_accuracy, validation_accuracy, test_accuracy, num_epochs):

        train_dataset = tf.data.Dataset.zip((X_train, Y_train))

        num_examples = train_dataset.cardinality().numpy()
        train_mini_batches = train_dataset.batch(self.minibatch_size).prefetch(8)

        for i in range(num_epochs):

            tic = time.time()
            epoch_cost_train = 0.0
            epoch_cost_val = 0.0
            epoch_cost_test = 0.0

            train_accuracy.reset_state()
            validation_accuracy.reset_state()
            test_accuracy.reset_state()

            for (X_minibatch, Y_minibatch) in train_mini_batches:
                with GradientTape() as tape:
                    ZL = self.forward_propagation(
                        X=tf.transpose(X_minibatch),
                        parameters=parameters
                    )
                    minibatch_cost = self.compute_cost(
                        ZL=ZL,
                        Y=tf.transpose(Y_minibatch)
                    )
                train_accuracy.update_state(Y_minibatch, tf.transpose(ZL))
                trainable = list(parameters.values())
                grads = tape.gradient(minibatch_cost, trainable)
                optimizer.apply_gradients(zip(grads, trainable))
                epoch_cost_train += minibatch_cost

            epoch_cost_train /= num_examples
            self.train_cost.append(epoch_cost_train.numpy())

            if X_validation is not None:
                validation_dataset = tf.data.Dataset.zip((X_validation, Y_validation))
                val_mini_batches = validation_dataset.batch(self.minibatch_size).prefetch(8)
                for (X_minibatch, Y_minibatch) in val_mini_batches:
                    ZL = self.forward_propagation(
                        X=tf.transpose(X_minibatch),
                        parameters=parameters
                    )
                    minibatch_cost = self.compute_cost(
                        ZL=ZL,
                        Y=tf.transpose(Y_minibatch)
                    )
                    epoch_cost_val += minibatch_cost
                    validation_accuracy.update_state(Y_minibatch, tf.transpose(ZL))
                epoch_cost_val /= num_examples
                self.validation_cost.append(epoch_cost_val.numpy())

            if X_test is not None:
                test_dataset = tf.data.Dataset.zip((X_test, Y_test))
                test_mini_batches = test_dataset.batch(self.minibatch_size).prefetch(8)
                for (X_minibatch, Y_minibatch) in test_mini_batches:
                    ZL = self.forward_propagation(
                        X=tf.transpose(X_minibatch),
                        parameters=parameters
                    )
                    minibatch_cost = self.compute_cost(
                        ZL=ZL,
                        Y=tf.transpose(Y_minibatch)
                    )
                    epoch_cost_test += minibatch_cost
                    test_accuracy.update_state(Y_minibatch, tf.transpose(ZL))
                epoch_cost_test /= num_examples
                self.test_cost.append(epoch_cost_test.numpy())

            toc = time.time()
            self.train_duration.append(toc - tic)

            if self.print_cost and i % self.num_print_cost == 0:
                print(f"Epoch: {i}: Cost: {epoch_cost_train}; Duration: {round(sum(self.train_duration[i - self.num_print_cost: i]), 5)}")
                print(f"Train Accuracy: {train_accuracy.result()}")

                if X_validation is not None:
                    print(f"Validation Accuracy: {validation_accuracy.result()}")

                if X_test is not None:
                    print(f"Test Accuracy: {test_accuracy.result()}")

        return parameters

    def model(self, X_train, Y_train, num_hidden_units, X_validation=None, Y_validation=None, X_test=None, Y_test=None , num_epochs=1000, learning_rate=1e-3):

        optimizer = None

        if self.optimizer == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif self.optimizer == "SGD":
            optimizer = SGD(learning_rate=learning_rate)

        train_accuracy = CategoricalAccuracy()
        validation_accuracy = CategoricalAccuracy()
        test_accuracy = CategoricalAccuracy()

        num_layer_units = self.layer_sizes(
            X=X_train,
            Y=Y_train,
            num_hidden_units=num_hidden_units
        )
        parameters = self.initialize_parameters(
            num_layer_units=num_layer_units,
            seed=self.seed
        )

        parameters = self.train(
            X_train=X_train,
            Y_train=Y_train,
            X_validation=X_validation,
            Y_validation=Y_validation,
            X_test=X_test,
            Y_test=Y_test,
            parameters=parameters,
            optimizer=optimizer,
            train_accuracy=train_accuracy,
            validation_accuracy=validation_accuracy,
            test_accuracy=test_accuracy,
            num_epochs=num_epochs
        )
        self.parameters = parameters

        if self.visualize_cost:
            plt.plot(self.train_cost, label="Train Cost")
            if X_validation is not None:
                plt.plot(self.validation_cost, label="Validation Cost")
            if X_test is not None:
                plt.plot(self.test_cost, label="Test Cost")
            plt.xlabel("# Epoch")
            plt.ylabel("Cost")
            plt.legend()
            plt.show()


train_dataset = h5py.File("/home/samani/Documents/projects/deep-learning/data/train_signs.h5")
test_dataset = h5py.File("/home/samani/Documents/projects/deep-learning/data/test_signs.h5")

X_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
Y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

X_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
Y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])


def one_hot_encoding(label, C=6):
    one_hot = tf.reshape(tf.one_hot(label, depth=C, axis=0), shape=[C, ])

    return one_hot


Y_train = Y_train.map(one_hot_encoding)
Y_test = Y_test.map(one_hot_encoding)


def normalize(image):

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1,])

    return image


X_train = X_train.map(normalize)
X_test = X_test.map(normalize)

obj = ImageRecognition(
    optimizer="Adam",
    initializer="Xavier",
    print_cost=True,
    visualize_cost=True,
    num_print_cost=20,
    minibatch_size=64,
    seed=1
)
obj.model(
    X_train=X_train,
    Y_train=Y_train,
    num_hidden_units=[20, 10, 8],
    X_test=X_test,
    Y_test=Y_test,
    num_epochs=400,
    learning_rate=1e-4
)
