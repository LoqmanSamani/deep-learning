import numpy as np
import tensorflow as tf


class RNN:
    """
    A simple RNN model class with methods to initialize parameters,
    perform forward and backward passes, and train the model.
    """
    def __init(self, epochs, num_hidden_units, learning_rate, weight_decay):

        self.epochs = epochs
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        """
        Initializes the RNN model parameters.

        Args:
            epochs (int): Number of epochs for training.
            num_hidden_units (int): Number of hidden units in the RNN.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        """




    def initialize_params(self, n_a, n_x, n_y, m):
        """
        Initializes the parameters of the RNN model.

        Args:
            n_a (int): Number of hidden units.
            n_x (int): Number of input features.
            n_y (int): Number of output units.
            m (int): Number of training examples (batch size).

        Returns:
            dict: A dictionary containing initialized weights and biases.
            tf.Variable: Initialized hidden state A0.
        """


        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        a_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=0)

        A0 = tf.Variable(a_initializer(shape=(n_a, m)), trainable=True)
        Wax = initializer(shape=(n_a, n_x))
        Waa = initializer(shape=(n_a, n_a))
        Wya = initializer(shape=(n_y, n_a))
        ba = initializer(shape=(n_a, 1))
        by = initializer(shape=(n_y, 1))

        parameters = {
            "Wax": Wax,
            "Waa": Waa,
            "Wya": Wya,
            "ba": ba,
            "by": by
        }

        return parameters, A0



    def compute_cost(self, Y_true, Y_hat, t_x):
        """
        Computes the cost function for the RNN model using categorical cross-entropy.

        Args:
            Y_true (tf.Tensor): Ground truth labels, shape (batch_size, num_classes, sequence_length).
            Y_hat (tf.Tensor): Predicted outputs, shape (batch_size, num_classes, sequence_length).
            t_x (int): Number of time steps (sequence length).

        Returns:
            tf.Tensor: Computed cost (mean cross-entropy loss).
        """

        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        Y_hat = tf.transpose(Y_hat, perm=[2, 1, 0])
        Y_true = tf.transpose(Y_true, perm=[2, 1, 0])

        loss = 0
        for t in range(t_x):
            loss += loss_fn(Y_true[t], Y_hat[t])

        loss = loss / t_x

        return loss


    def cell_forward(self, X, A, params):
        """
        Performs a forward pass for a single RNN cell.

        Args:
            X (tf.Tensor): Input at the current time step, shape (n_x, batch_size).
            A (tf.Tensor): Hidden state from the previous time step, shape (n_a, batch_size).
            params (dict): Dictionary containing the model parameters.

        Returns:
            tuple: A tuple containing:
                - A_ (tf.Tensor): Updated hidden state, shape (n_a, batch_size).
                - Y_hat (tf.Tensor): Output prediction, shape (n_y, batch_size).
                - cache (tuple): Cache containing intermediate values for backpropagation.
        """

        WA = tf.linalg.matmul(a=params["Waa"], b=A, name="WA forward")
        WX = tf.linalg.matmul(a=params["Wax"], b=X, name="WX forward")

        A_ = tf.math.tanh(WX + WA + params["ba"], name="A_ forward")

        WY = tf.linalg.matmul(a=params["Wya"], b=A_, name="WY forward")
        Y_hat = tf.keras.activations.softmax(x=WY + params["by"], axis=0)

        cache = (A_, A, X, params)

        return A_, Y_hat, cache

    def cells_forward(self, X, A0, params):
        """
        Performs a forward pass through the entire RNN sequence.

        Args:
            X (tf.Tensor): Input data, shape (n_x, batch_size, sequence_length).
            A0 (tf.Variable): Initial hidden state, shape (n_a, batch_size).
            params (dict): Dictionary containing the model parameters.

        Returns:
            tuple: A tuple containing:
                - A (tf.Tensor): Hidden states over the sequence, shape (n_a, batch_size, sequence_length).
                - Y_hat (tf.Tensor): Output predictions over the sequence, shape (n_y, batch_size, sequence_length).
                - caches (tuple): A tuple containing the input data and list of caches for each time step.
        """

        caches = []
        n_x, m, t_x = X.shape

        A = tf.TensorArray(
            dtype=tf.float32,
            size=t_x,
            clear_after_read=False
        )
        Y_hat = tf.TensorArray(
            dtype=tf.float32,
            size=t_x,
            clear_after_read=False
        )

        A_ = A0

        for t in range(t_x):

            A_, y_hat, cache = self.cell_forward(
                X=X[:, :, t],
                A=A_,
                params=params
            )

            A = A.write(t, A_)
            Y_hat = Y_hat.write(t, y_hat)

            caches.append(cache)

        A = tf.transpose(A.stack(), perm=[1, 2, 0])
        Y_hat = tf.transpose(Y_hat.stack(), perm=[1, 2, 0])

        caches = (X, caches)

        return A, Y_hat, caches





    def rrn_model(self, X, Y):
        """
        Trains the RNN model using the given input data and labels.

        Args:
            X (np.ndarray): Input data, shape (n_x, batch_size, sequence_length).
            Y (np.ndarray): Ground truth labels, shape (batch_size, n_y, sequence_length).

        Returns:
            tuple: A tuple containing:
                - params (dict): Trained model parameters.
                - costs (list): List of cost values recorded during training.
        """


        costs = []
        Y = tf.convert_to_tensor(Y)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        n_x, m, t_x = X.shape
        _, n_y = Y.shape
        n_a = self.num_hidden_units

        params, A0 = self.initialize_params(
            n_a=n_a,
            n_x=n_x,
            n_y=n_y,
            m=m
        )

        for i in range(self.epochs):
            with tf.GradientTape() as tape:

                A0.assign(tf.zeros_like(A0))

                A, Y_hat, caches = self.cells_forward(
                    X=X,
                    A0=A0,
                    params=params
                )
                cost = self.compute_cost(
                    Y_true=Y,
                    Y_hat=Y_hat,
                    t_x=t_x
                )
                costs.append(cost)

            print(f"Epoch {i + 1}/{self.epochs}, Cost: {cost}")

            variables = list(params.values()) + [A0]
            gradients = tape.gradient(cost, variables)
            optimizer.apply_gradients(zip(gradients, variables))


        return params, costs











