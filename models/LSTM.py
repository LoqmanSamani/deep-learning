import tensorflow as tf




class LSTM:
    def __init__(self, epochs, num_hidden_units, learning_rate, weight_decay):

        """
        Implements a Long Short-Term Memory (LSTM) network for sequence prediction.

        This class encapsulates the core components of an LSTM network, including the initialization of parameters,
        forward propagation through LSTM cells, and training with gradient descent. The LSTM model is used for sequence
        prediction tasks where the input is a sequence of vectors, and the output is a sequence of probability distributions
        over classes.

        Attributes:
            epochs (int): Number of training epochs.
            num_hidden_units (int): Number of hidden units in the LSTM layers.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay parameter for regularization.
            params (dict): Dictionary of model parameters after training.
            cost_ (list): List of cost values for each epoch during training.
        """

        self.epochs = epochs
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.params = None
        self.cost_ = None

        """
        Initializes the LSTM model with hyperparameters.

        Args:
            epochs (int): Number of training epochs.
            num_hidden_units (int): Number of hidden units in the LSTM layers.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay parameter for regularization.
        """




    def initialize_params(self, n_a, n_x, n_y, m):

        """
        Initializes the parameters of the LSTM model.

        Args:
            n_a (int): Number of hidden units in the LSTM layers.
            n_x (int): Number of input features.
            n_y (int): Number of output classes.
            m (int): Number of examples (batch size).

        Returns:
            params (dict): Dictionary of initialized parameters including weights and biases.
            a0 (tf.Variable): Initial hidden state.
            c0 (tf.Variable): Initial cell state.
        """


        initializer = tf.keras.initializers.GlorotNormal(seed=0)

        # Initialize hidden and cell states as Variables
        a0 = tf.Variable(tf.zeros(shape=(n_a, m), dtype=tf.float32), trainable=False)
        c0 = tf.Variable(tf.zeros(shape=(n_a, m), dtype=tf.float32), trainable=False)

        # Weight matrices for LSTM gates
        Wf = initializer(shape=(n_a, n_a + n_x))  # Forget gate weights
        Wu = initializer(shape=(n_a, n_a + n_x))  # Update (input) gate weights
        Wc = initializer(shape=(n_a, n_a + n_x))  # Candidate cell state weights
        Wo = initializer(shape=(n_a, n_a + n_x))  # Output gate weights

        # Biases for LSTM gates (Forget gate bias initialized to 1)
        bf = tf.ones((n_a, 1), dtype=tf.float32)  # Forget gate bias
        bu = tf.zeros((n_a, 1), dtype=tf.float32)  # Update gate bias
        bc = tf.zeros((n_a, 1), dtype=tf.float32)  # Candidate state bias
        bo = tf.zeros((n_a, 1), dtype=tf.float32)  # Output gate bias

        # Output layer weights and biases
        Wy = initializer(shape=(n_y, n_a))  # Output layer weights
        by = tf.zeros((n_y, 1), dtype=tf.float32)  # Output layer bias

        # Dictionary of all parameters
        params = {
            "Wf": Wf, "bf": bf,
            "Wu": Wu, "bu": bu,
            "Wc": Wc, "bc": bc,
            "Wo": Wo, "bo": bo,
            "Wy": Wy, "by": by
        }

        return params, a0, c0




    def compute_cost(self, y_true, y_hat, t_x):
        """
        Computes the loss between the true labels and predicted labels.

        Args:
            y_true (tf.Tensor): True labels, shaped as (n_y, m).
            y_hat (tf.Tensor): Predicted labels, shaped as (n_y, m).
            t_x (int): Number of time steps in the sequence.

        Returns:
            loss (tf.Tensor): Computed loss value.
        """

        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        loss = loss_fn(y_true, y_hat)

        return loss




    def cell_forward(self, x, c_prev, a_prev, params):
        """
        Performs forward propagation through a single LSTM cell.

        Args:
            x (tf.Tensor): Input data for the current time step.
            c_prev (tf.Tensor): Previous cell state.
            a_prev (tf.Tensor): Previous hidden state.
            params (dict): Dictionary of model parameters including weights and biases.

        Returns:
            a_next (tf.Tensor): Updated hidden state.
            c_next (tf.Tensor): Updated cell state.
            y_pred (tf.Tensor): Predicted output for the current time step.
            cache (tuple): Cache of intermediate values needed for backpropagation.
        """

        forget_gate = tf.keras.activations.sigmoid(tf.linalg.matmul(params["Wf"], tf.concat(values=[a_prev, x], axis=0)) + params["bf"])
        update_gate = tf.keras.activations.sigmoid(tf.linalg.matmul(params["Wu"], tf.concat(values=[a_prev, x], axis=0)) + params["bu"])
        cell_state = tf.math.tanh(tf.linalg.matmul(params["Wc"], tf.concat(values=[a_prev, x], axis=0)) + params["bc"])
        c_next = forget_gate * c_prev + update_gate * cell_state
        output_gate = tf.keras.activations.sigmoid(tf.linalg.matmul(params["Wo"], tf.concat(values=[a_prev, x], axis=0)) + params["bo"])
        a_next = output_gate * tf.math.tanh(c_next)
        y_pred = tf.keras.activations.softmax(tf.linalg.matmul(params["Wy"], a_next) + params["by"])

        cache = (a_next, c_next, a_prev, c_prev, forget_gate, update_gate, cell_state, output_gate, x, params)

        return a_next, c_next, y_pred, cache



    def cells_forward(self, x, a0, params):
        """
        Performs forward propagation through all LSTM cells over the sequence.

        Args:
            x (tf.Tensor): Input data for all time steps, shaped as (n_x, m, t_x).
            a0 (tf.Variable): Initial hidden state.
            params (dict): Dictionary of model parameters including weights and biases.

        Returns:
            a (tf.Tensor): Hidden states for all time steps, shaped as (n_a, m, t_x).
            c (tf.Tensor): Cell states for all time steps, shaped as (n_a, m, t_x).
            y (tf.Tensor): Predicted outputs for all time steps, shaped as (n_y, m, t_x).
            caches (tuple): Cache of all intermediate values needed for backpropagation.
        """

        caches = []
        n_x, m, t_x = x.shape
        n_y, n_a = params["Wy"].shape

        a = tf.TensorArray(dtype=tf.float32, size=t_x, clear_after_read=False)
        c = tf.TensorArray(dtype=tf.float32, size=t_x, clear_after_read=False)
        y = tf.TensorArray(dtype=tf.float32, size=t_x, clear_after_read=False)

        a_next = a0
        c_next = tf.zeros(shape=(n_a, m), dtype=tf.float32)

        for t in range(t_x):

            xt = x[:, :, t]
            a_next, c_next, yt, cache = self.cell_forward(
                x=xt,
                c_prev=c_next,
                a_prev=a_next,
                params=params
            )
            a = a.write(t, a_next)
            c = c.write(t, c_next)
            y = y.write(t, yt)

            caches.append(cache)

        a = tf.transpose(a.stack(), perm=[1, 2, 0])
        c = tf.transpose(c.stack(), perm=[1, 2, 0])
        y = tf.transpose(y.stack(), perm=[1, 2, 0])
        caches = (caches, x)

        return a, c, y, caches


    def lstm_model(self, x, y):
        """
        Trains the LSTM model using the provided input data and labels.

        Args:
            x (tf.Tensor): Input data for all time steps, shaped as (n_x, m, t_x).
            y (tf.Tensor): True labels, shaped as (n_y, m).

        Returns:
            params (dict): Dictionary of trained model parameters.
            cost (list): List of cost values for each epoch during training.
        """

        y = tf.convert_to_tensor(y)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        n_x, m, t_x = x.shape
        _, n_y = y.shape
        n_a = self.num_hidden_units

        params, a0, c0 = self.initialize_params(
            n_a=n_a,
            n_x=n_x,
            n_y=n_y,
            m=m
        )

        for i in range(self.epochs):

            with tf.GradientTape() as tape:

                a0.assign(tf.zeros_like(a0))
                c0.assign(tf.zeros_like(c0))

                a, c, y_hat, cache = self.cells_forward(
                    x=x,
                    a0=a0,
                    params=params
                )

                cost = self.compute_cost(
                    y_true=y,
                    y_hat=y_hat,
                    t_x=t_x
                )
                self.cost_.append(cost)

            print(f"Epoch {i + 1}/{self.epochs}, Cost: {cost}")

            variables = list(params.values())
            gradients = tape.gradient(cost, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        self.params = params

        return self.params, self.cost_



