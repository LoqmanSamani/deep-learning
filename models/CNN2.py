import numpy as np
import time
import h5py


class Conv2DForward:
    def __init__(self):
        """
        Initializes Conv2D object with default values.
        """
        pass

    def initialize_parameters(self, f, num_channels, filters):
        """
        Initializes the weight matrix W and bias vector b for convolutional filters.

        Args:
            - f (int): Size of the filters (kernel size).
            - num_channels (int): Number of channels in the input data.
            - filters (int): Number of filters.

        Returns:
            - tuple: A tuple containing the initialized weight matrix W and bias vector b.
        """

        W = np.random.randn(filters, f, f, num_channels) * np.sqrt(2 / f)
        b = np.random.randn(filters, 1, 1, 1) * np.sqrt(2 / f)

        return (W, b)

    def zero_padding(self, X, padding):
        """
        Applies zero-padding to the input data.

        Args:
            - X (ndarray): Input data with shape (m, n_H, n_W, n_C).
            - padding (int): Number of padding columns or rows.

        Returns:
            - X_pad (ndarray): Input data with zero-padding applied.
        """

        X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant", constant_values=(0, 0))

        return X_pad

    def x_conv_shape(self, n_pre, f, s):
        """
        Calculates the output size of the convolution operation along a single dimension.

        Args:
            n_pre (int): Size of the input along a particular dimension (height or width).
            f (int): Size of the filter (kernel).
            s (int): Stride.

        Returns:
            n (int): Output size of the convolution operation.
        """

        n = int((n_pre - f / s)) + 1

        return n

    def relu(self, Z):
        """
        Applies the ReLU activation function element-wise.

        Args:
            Z (ndarray): Input data.

        Returns:
            tuple: A tuple containing the output data (activation) and the input data (cache).
        """

        A = np.maximum(0, Z)
        cache = Z

        return A, cache

    def conv_1_filter(self, X, W, nh, nw, stride):
        """
        Applies convolution with a single filter to the input data.

        Args:
            - X (ndarray): Input data with shape (nh, nw, n_C).
            - W (ndarray): Filter weights with shape (f, f, n_C).
            - nh (int): Output size along the height dimension.
            - nw (int): Output size along the width dimension.
            - stride (int): Stride.

        Returns:
            - X_conv (ndarray): Output of the convolution operation.
        """

        (f, f, n_c) = W.shape
        X_conv = np.zeros((nh, nw))

        for k in range(n_c):

            for i in range(nh):
                v_start = i * stride
                v_end = v_start + f
                for j in range(nw):
                    h_start = j * stride
                    h_end = h_start + f

                    frame = X[v_start:v_end, h_start:h_end, k]
                    X_conv[i, j] += np.sum(frame * W[:, :, k])

        return X_conv

    def conv2d(self, X, filters, kernel_size=(3, 3), padding=2, strides=1, W=None, b=None):
        """
        Applies 2D convolution operation to the input data.

        Args:
            - X (ndarray): Input data with shape (m, n_h, n_W, n_C).
            - filters (int): Number of applied filters.
            - kernel_size (tuple): Size of the applied filters (f, f).
            - padding (int): Number of padding columns or rows.
            - strides (int): Number of strides.
            - W (ndarray): weight matrix W.
            - b (ndarray): bias array

        Returns:
            tuple: A tuple containing the output data, cache, weight matrix W, and bias vector b.
        """

        num_channels = X[0].shape[-1]
        f = kernel_size[0]

        if W is None:
            (W, b) = self.initialize_parameters(f=f, num_channels=num_channels, filters=filters)

        linear_cache = (X, W, b, filters, kernel_size, padding, strides)
        X_pad = self.zero_padding(X=X, padding=padding)

        nh = self.x_conv_shape(n_pre=X_pad[0].shape[0], f=f, s=strides)
        nw = self.x_conv_shape(n_pre=X_pad[0].shape[1], f=f, s=strides)

        m = X_pad.shape[0]

        X_conv = np.zeros((m, nh, nw, filters))
        activation_cache = np.zeros((m, nh, nw, filters))

        for i in range(m):
            X_temp = np.zeros((nh, nw, filters))

            for f in range(filters):
                Z = self.conv_1_filter(X=X_pad[i, :, :, :], W=W[f, :, :, :], nh=nh, nw=nw, stride=strides) + b[f]
                X_temp[:, :, f], activation_cache[i, :, :, f] = self.relu(Z=Z)

            X_conv[i, :, :, :] = X_temp

        caches = (linear_cache, activation_cache)

        return (X_conv, caches, W, b)





class Conv2DBackward:
    """
    Class implementing backward propagation for a convolutional layer.

    Methods:
        zero_padding(X, padding):
            Apply zero padding to the input.

        backward_relu(dA, Z):
            Compute the backward pass for the ReLU activation function.

        conv_1_backward(Al, dAl_1, W, dZ, dW, db, step, nh, nw, nc, stride, f):
            Compute gradients for a single convolution operation.

        conv2d_backward(dAl, caches):
            Perform backward propagation for the entire convolutional layer.
    """
    def __init__(self):
        pass

    def zero_padding(self, X, padding):
        """
        Apply zero padding to the input.

        Args:
            - X (numpy.ndarray): Input data of shape (m, n_H, n_W, n_C).
            - padding (int): Number of padding columns or rows.

        Returns:
            - X_pad (numpy.ndarray): Padded input data of shape (m, n_H_pad, n_W_pad, n_C).
        """

        X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant", constant_values=(0, 0))

        return X_pad

    def backward_relu(self, dA, Z):
        """
        Compute the backward pass for the ReLU activation function.

        Args:
            - dA (numpy.ndarray): Gradient of the cost with respect to the post-activation values.
            - Z (numpy.ndarray): Pre-activation output.

        Returns:
            - dZ (numpy.ndarray): Gradient of the cost with respect to Z.
                """

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

    def conv_1_backward(self, Al, dAl_1, W, dZ, dW, db, step, nh, nw, nc, stride, f):
        """
        Compute gradients for a single convolution operation.

        Args:
            - Al (numpy.ndarray): Input slice to the convolution operation.
            - dAl_1 (numpy.ndarray): Gradient of the cost with respect to the input Al.
            - W (numpy.ndarray): Weights of the convolution operation.
            - dZ (numpy.ndarray): Gradient of the cost with respect to the output Z.
            - dW (numpy.ndarray): Gradient of the cost with respect to the weights W.
            - db (numpy.ndarray): Gradient of the cost with respect to the biases.
            - step (int): Step index.
            - nh (int): Height of the output feature map.
            - nw (int): Width of the output feature map.
            - nc (int): Number of channels.
            - stride (int): Stride of the convolution operation.
            - f (int): Filter size.

        Returns:
            - tuple: Tuple containing gradients of the cost with respect to Al, W, and b.
                """

        for h in range(nh):
            for w in range(nw):
                for c in range(nc):
                    v_start = h * stride
                    v_end = v_start + f
                    h_start = w * stride
                    h_end = h_start + f

                    a_slice = Al[v_start:v_end, h_start:h_end, :]

                    dAl_1[v_start:v_end, h_start:h_end, :] += W[c, :, :, :] * dZ[step, h, w, c]
                    dW[c, :, :, :] += a_slice * dZ[step, h, w, c]
                    db[c, :, :, :] += dZ[step, h, w, c]

        return (dAl_1, dW, db)

    def conv2d_backward(self, dAl, caches):
        """
        Perform backward propagation for the entire convolutional layer.

        Args:
            - dAl (numpy.ndarray): Gradient of the cost with respect to the activations of the current layer.
            - caches (tuple): Tuple of cache values from the forward pass. It contains:
                - linear_cache (tuple): Tuple of values from the linear (convolutional) part of the forward pass.
                    It includes:
                        - X (numpy.ndarray): Input data of shape (m, n_H_prev, n_W_prev, n_C_prev).
                        - W (numpy.ndarray): Weights of the convolutional operation of shape (f, f, n_C_prev, n_C).
                        - b (numpy.ndarray): Biases of the convolutional operation of shape (1, 1, 1, n_C).
                        - filters (int): Number of applied filters.
                        - kernel_size (tuple): Size of the applied filters (f, f).
                        - padding (int): Number of padding columns or rows.
                        - strides (int): Number of strides.
                        - layer (int): Layer number.
                - activation_cache (numpy.ndarray): Activation values from the forward pass,
                    which are the input to the activation function (ReLU) of shape (m, n_H, n_W, n_C).

        Returns:
            - tuple: Tuple containing gradients of the cost with respect to the activations, weights, and biases.
                - dAl_1 (numpy.ndarray): Gradient of the cost with respect to the activations of the previous layer,
                    of shape (m, n_H_prev, n_W_prev, n_C_prev).
                - dW (numpy.ndarray): Gradient of the cost with respect to the weights of the convolutional layer,
                    of shape (f, f, n_C_prev, n_C).
                - db (numpy.ndarray): Gradient of the cost with respect to the biases of the convolutional layer,
                    of shape (1, 1, 1, n_C).
        """

        (linear_cache, activation_cache) = caches
        (X, W, b, filters, kernel_size, padding, strides) = linear_cache

        Z = activation_cache

        dZ = self.backward_relu(
            dA=dAl,
            Z=Z
        )

        (m, nh_pre, nw_pre, nc_pre) = X.shape
        (nc, f, f, nc_pre) = W.shape
        (m, nh, nw, nc) = dZ.shape

        dAl_1 = np.zeros(X.shape)
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)

        # Pad X and dAl_1
        X_pad = self.zero_padding(
            X=X,
            padding=padding
        )
        dAl_1_pad = self.zero_padding(
            X=dAl_1,
            padding=padding
        )

        for i in range(m):

            x_pad = X_pad[i]
            dal_1_pad = dAl_1_pad[i]

            (dal_1_pad, dW, db) = self.conv_1_backward(
                Al=x_pad,
                dAl_1=dal_1_pad,
                W=W,
                dZ=dZ,
                dW=dW,
                db=db,
                step=i,
                nh=nh,
                nw=nw,
                nc=nc,
                stride=strides,
                f=f
            )

        # Remove padding from dAl_1_pad
        dAl_1 = dAl_1_pad[:, padding:-padding, padding:-padding, :]
        assert (dAl_1.shape == (m, nh_pre, nw_pre, nc_pre))

        return (dAl_1, dW, db)






class Pool2DForward:
    """
    This class implements 2D pooling operations, including both max pooling and average pooling.

    Methods:

        compute_output_size(nh_pre, nw_pre, nc_pre, f, stride):
            Computes the output dimensions for the pooling operation.

        pool_max_forward(A, f, stride):
            Performs forward pass of max pooling operation.

        pool_average_forward(A, f, stride):
            Performs forward pass of average pooling operation.
    """

    def __init__(self):
        """
        Initialize the Pool2DForward class.
        """
        pass



    def compute_output_size(self, nh_pre, nw_pre, nc_pre, f, stride):

        """
        Computes the dimensions of the output after applying the pooling operation.

        Parameters:

            - nh_pre (int): Height of the input volume.
            - nw_pre (int): Width of the input volume.
            - nc_pre (int): Number of channels in the input volume.
            - f (int): Size of the pooling window.
            - stride (int): Stride with which the pooling window moves.

        Returns:
            - tuple : A tuple (nh, nw, nc) representing the height, width, and number of channels of the output volume.
        """

        nh = int(1 + (nh_pre - f) / stride)
        nw = int(1 + (nw_pre - f) / stride)
        nc = nc_pre

        return (nh, nw, nc)

    def pool_max_forward(self, A, f, stride):
        """
        Performs the forward pass of the max pooling operation.

        Parameters:

            - A (numpy.ndarray): Input data of shape (m, nh_pre, nw_pre, nc_pre).
            - f (int): Size of the pooling window.
            - stride (int): Stride with which the pooling window moves.

        Returns:

            - tuple: A tuple (A_pool, cache) where A_pool is the output of the max pooling operation
                     and cache contains the values needed for the backward pass.
        """

        (m, nh_pre, nw_pre, nc_pre) = A.shape

        (nh, nw, nc) = self.compute_output_size(
            nh_pre=nh_pre,
            nw_pre=nw_pre,
            nc_pre=nc_pre,
            f=f,
            stride=stride
        )

        A_pool = np.zeros((m, nh, nw, nc))

        for i in range(m):

            for h in range(nh):

                v_start = h * stride
                v_end = v_start + f

                for w in range(nw):

                    h_start = w * stride
                    h_end = h_start + f

                    for c in range(nc):
                        a_slice = A[i, v_start:v_end, h_start:h_end, c]
                        A_pool[i, h, w, c] = np.max(a_slice, axis=(0, 1))

        cache = (A, f, stride)

        return (A_pool, cache)

    def pool_average_forward(self, A, f, stride):

        """
        Performs the forward pass of the average pooling operation.

        Parameters:

            - A (numpy.ndarray): Input data of shape (m, nh_pre, nw_pre, nc_pre).
            - f (int): Size of the pooling window.
            - stride (int): Stride with which the pooling window moves.

        Returns:

            - tuple: A tuple (A_pool, cache) where A_pool is the output of the average pooling operation
                     and cache contains the values needed for the backward pass.
        """

        (m, nh_pre, nw_pre, nc_pre) = A.shape
        (nh, nw, nc) = self.compute_output_size(
            nh_pre=nh_pre,
            nw_pre=nw_pre,
            nc_pre=nc_pre,
            f=f,
            stride=stride
        )

        A_pool = np.zeros((m, nh, nw, nc))

        for i in range(m):

            for h in range(nh):

                v_start = h * stride
                v_end = v_start + f

                for w in range(nw):

                    h_start = w * stride
                    h_end = h_start + f

                    for c in range(nc):
                        a_slice = A[i, v_start:v_end, h_start:h_end, c]
                        A_pool[i, h, w, c] = np.mean(a_slice, axis=(0, 1))

        cache = (A, f, stride)

        return (A_pool, cache)





class Pool2DBackward:

    """
    This class implements the backward pass of 2D pooling operations, including both max pooling and average pooling.

    Methods:
        distribute_value(dZ, shape):
            Distributes the input value evenly across a given shape.

        pool_max_backward(dA, cache):
            Performs backward pass of the max pooling operation.

        pool_average_backward(dA, cache):
            Performs backward pass of the average pooling operation.
    """

    def __init__(self):
        """
        Initialize the Pool2DBackward class.
        """
        pass

    def distribute_value(self, dZ, shape):

        """
        Distributes the input value evenly across a given shape.

        Parameters:

            - dZ (float): The input value to distribute.
            - shape (tuple): The shape (height, width) over which to distribute the value.

        Returns:

            - A (numpy.ndarray): An array of the specified shape with the input value evenly distributed.
        """

        (nh, nw) = shape
        average = dZ / (nh * nw)
        A = np.ones(shape) * average

        return A



    def pool_max_backward(self, dA, cache):

        """
        Performs the backward pass of the max pooling operation.

        Parameters:

            - dA (numpy.ndarray): Gradient of the cost with respect to the output of the pooling layer,
                                  same shape as the output of the pooling layer (m, nh, nw, nc).
            - cache (tuple): A tuple of (A, f, stride) where A is the input data,
                             f is the pooling window size, and stride is the stride of the pooling window.

        Returns:

        dA_pre (numpy.ndarray): Gradient of the cost with respect to the input of the pooling layer,
                                same shape as the input (m, nh_pre, nw_pre, nc_pre).
        """

        (A, f, stride) = cache

        m, nh, nw, nc = dA.shape

        dA_pre = np.zeros(A.shape)

        for i in range(m):

            a_pre = A[i]

            for h in range(nh):
                for w in range(nw):
                    for c in range(nc):
                        v_start = h * stride
                        v_end = v_start + f
                        h_start = w * stride
                        h_end = h_start + f

                        a_slice = a_pre[v_start: v_end, h_start: h_end, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_pre[i, v_start: v_end, h_start: h_end, c] += mask * dA[i, h, w, c]

        return dA_pre

    def pool_average_backward(self, dA, cache):

        """
        Performs the backward pass of the average pooling operation.

        Parameters:

            - dA (numpy.ndarray): Gradient of the cost with respect to the output of the pooling layer,
                                  same shape as the output of the pooling layer (m, nh, nw, nc).
            - cache (tuple): A tuple of (A, f, stride) where A is the input data, f is the pooling window size,
                             and stride is the stride of the pooling window.

        Returns:

            - dA_pre (numpy.ndarray): Gradient of the cost with respect to the input of the pooling layer,
                                      same shape as the input (m, nh_pre, nw_pre, nc_pre).
        """

        (A, f, stride) = cache

        m, nh, nw, nc = dA.shape

        dA_pre = np.zeros(A.shape)

        for i in range(m):

            for h in range(nh):
                for w in range(nw):
                    for c in range(nc):
                        v_start = h * stride
                        v_end = v_start + f
                        h_start = w * stride
                        h_end = h_start + f

                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_pre[i, v_start: v_end, h_start: h_end, c] += self.distribute_value(
                            dZ=da,
                            shape=shape
                        )

        return dA_pre


class DenseFroward:
    """
    This class implements the forward pass for a fully connected (dense) layer, including
    initialization of parameters and activation functions such as ReLU, Sigmoid, and Softmax.

    Methods:

        - initialize_parameters(A, units, seed):
          Initializes weights and biases for the dense layer.
        - relu(Z):
            Applies the ReLU activation function.
        - sigmoid(Z):
            Applies the Sigmoid activation function.
        - softmax(Z):
            Applies the Softmax activation function.
        - fc_forward(A, units, activation="relu", W=None, b=None, seed=0):
            Performs the forward pass through the dense layer.
    """

    def __init__(self):

        """
        Initializes the DenseForward class.
        """

        pass

    def initialize_parameters(self, A, units, seed):

        """
        Initializes weights and biases for a dense layer.

        Parameters:
            - A (numpy.ndarray): Flattened output of the last convolutional layer.
            - units (int): Number of neurons in the dense layer.
            - seed (int): Random seed for reproducibility.

        Returns:
            - tuple: Tuple containing the initialized weights (W) and biases (b).
        """

        np.random.seed(seed)

        W = np.random.randn(units, len(A)) * np.sqrt(2 / len(A))
        b = np.zeros((units, 1))

        return (W, b)


    def relu(self, Z):

        """
        ReLU activation function.

        Parameters:
            - Z (numpy.ndarray): Linear output of the layer.

        Returns:
            - tuple: A tuple (A, Z) where A is the activated output and Z is the input to the activation function.
        """

        A = np.maximum(0, Z)
        cache = Z

        return (A, cache)

    def sigmoid(self, Z):

        """
        Sigmoid activation function.

        Parameters:
            - Z (numpy.ndarray): Linear output of the layer.

        Returns:
            - tuple: A tuple (A, Z) where A is the activated output and Z is the input to the activation function.
        """

        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return (A, cache)

    def softmax(self, Z):

        """
        Softmax activation function.

        Parameters:
            - Z (numpy.ndarray): Linear output of the layer.

        Returns:
            - tuple: A tuple (A, Z) where A is the activated output and Z is the input to the activation function.
        """

        A = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 1e-15)

        cache = Z

        return (A, cache)

    def fc_forward(self, A, units, activation="relu", W=None, b=None, seed=0):

        """
        Forward pass through a fully connected (dense) layer.

        Parameters:
            - A (numpy.ndarray): Input data.
            - units (int): Number of neurons in the dense layer.
            - activation (str): Activation function to use ("relu", "sigmoid", or "softmax").
            - W (numpy.ndarray): Weights (optional, for reusability).
            - b (numpy.ndarray): Biases (optional, for reusability).
            - seed (int): Random seed for initialization (if W and b are not provided).

        Returns:
            - tuple: A tuple (A, W, b, cache) where A is the activated output, W is the weights,
                     b is the biases, and cache contains intermediate values for backpropagation.
        """

        if W is None:
            (W, b) = self.initialize_parameters(
                A=A,
                units=units,
                seed=seed
            )

        linear_cache = (A, W, b)

        Z = np.dot(W, A) + b

        if activation == "relu":

            (A, activation_cache) = self.relu(Z=Z)

        elif activation == "sigmoid":

            (A, activation_cache) = self.sigmoid(Z=Z)

        elif activation == "softmax":

            (A, activation_cache) = self.softmax(Z=Z)

        cache = (linear_cache, activation_cache)

        return (A, W, b, cache)




class DenseBackward:
    """
    This class implements the backward pass for a fully connected (dense) layer, including
    gradients for activation functions such as ReLU, Sigmoid, and Softmax.

    Methods:

        - backward_relu(dA, Z):
            Computes the gradient of the ReLU activation function.
        - backward_sigmoid(dA, Z):
            Computes the gradient of the Sigmoid activation function.
        - backward_softmax(dA, Z):
            Computes the gradient of the Softmax activation function.
        - fc_backward(A, Y, cache, activation, dA=None, last_layer=False):
            Performs the backward pass through the dense layer.
    """
    def __init__(self):
        """
        Initializes the DenseBackward class.
        """
        pass

    def backward_relu(self, dA, Z):
        """
        Computes the gradient of the ReLU activation function.

        Parameters:
            - dA (numpy.ndarray): Gradient of the loss with respect to the activation output.
            - Z (numpy.ndarray): Linear output of the layer before activation.

        Returns:
            - dZ (numpy.ndarray): Gradient of the loss with respect to Z.
        """

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

    def backward_sigmoid(self, dA, Z):
        """
        Computes the gradient of the Sigmoid activation function.

        Parameters:
            - dA (numpy.ndarray): Gradient of the loss with respect to the activation output.
            - Z (numpy.ndarray): Linear output of the layer before activation.

        Returns:
            - dZ (numpy.ndarray): Gradient of the loss with respect to Z.
        """

        A = 1 / (1 + np.exp(-Z))
        dZ = np.multiply(np.multiply(dA, A), 1 - A)

        return dZ

    def backward_softmax(self, dA, Z):
        """
        Computes the gradient of the Softmax activation function.

        Parameters:
            - dA (numpy.ndarray): Gradient of the loss with respect to the activation output.
            - Z (numpy.ndarray): Linear output of the layer before activation.

        Returns:
            - dZ (numpy.ndarray): Gradient of the loss with respect to Z.
        """

        A = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 1e-15)
        dZ = A * (1 - A) * dA

        return dZ

    def fc_backward(self, A, Y, cache, activation, dA=None, last_layer=False):
        """
        Performs the backward pass through a fully connected (dense) layer.

        Parameters:
            - A (numpy.ndarray): Activated output from the forward pass.
            - Y (numpy.ndarray): True labels.
            - cache (tuple): Cached values from the forward pass.
            - activation (str): Activation function used ("relu", "sigmoid", or "softmax").
            - dA (numpy.ndarray, optional): Gradient of the loss with respect to the activation output.
            - last_layer (bool): Whether this is the last layer of the network.

        Returns:
            - tuple: Gradients with respect to A_prev, W, and b (dA_prev, dW, db).
                """

        if last_layer:
            dA = - (np.divide(Y, A + 1e-15) - np.divide(1 - Y, 1 - A + 1e-15))

        (linear_cache, activation_cache) = cache
        (A, W, b) = linear_cache
        Z = activation_cache
        m = A.shape[1]

        if activation == "sigmoid":
            dZ = self.backward_sigmoid(dA=dA, Z=Z)

        elif activation == "softmax":
            dZ = self.backward_softmax(dA=dA, Z=Z)

        elif activation == "relu":
            dZ = self.backward_relu(dA=dA, Z=Z)

        dA = np.dot(W.T, dZ)
        dW = 1 / m * np.dot(dZ, A.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        return (dA, dW, db)



class UpdateAdam:
    """
    This class implements the Adam optimization algorithm for updating parameters
    of a neural network layer, including initialization of Adam parameters and
    computation of first and second momentums.

    Methods:
        - initialize_adam_parameters(W, b):
            Initializes Adam parameters (vdW, vdb, sdW, sdb).
        - compute_first_momentum(dW, db, vdW, vdb, beta, t):
            Computes the first momentum for Adam optimization.
        - compute_second_momentum(dW, db, sdW, sdb, beta, t):
            Computes the second momentum for Adam optimization.
        - update_parameters(dW, db, W, b, learning_rate, beta1, beta2, epsilon, t, adam_params=None):
            Updates parameters using the Adam optimization algorithm.
    """
    def __init__(self):
        """
        Initializes the UpdateAdam class.
        """
        pass

    def initialize_adam_parameters(self, W, b):
        """
        Initializes Adam parameters (vdW, vdb, sdW, sdb) to zeros.

        Parameters:
            - W (numpy.ndarray): Weights of the layer.
            - b (numpy.ndarray): Biases of the layer.

        Returns:
            - tuple: Tuple containing initialized vdW, vdb, sdW, sdb.
        """

        vdW = np.zeros(W.shape)
        vdb = np.zeros(b.shape)

        sdW = np.zeros(W.shape)
        sdb = np.zeros(b.shape)

        return (vdW, vdb, sdW, sdb)


    def compute_first_momentum(self, dW, db, vdW, vdb, beta, t):
        """
        Computes the first momentum for Adam optimization.

        Parameters:
            - dW (numpy.ndarray): Gradient of the loss with respect to weights.
            - db (numpy.ndarray): Gradient of the loss with respect to biases.
            - vdW (numpy.ndarray): First moment estimate for weights.
            - vdb (numpy.ndarray): First moment estimate for biases.
            - beta (float): Exponential decay rate for the first moment estimates.
            - t (int): Time step.

        Returns:
            - tuple: Tuple containing updated vdW and vdb.
        """

        vdW = (beta * vdW) + ((1 - beta) * dW)
        vdb = (beta * vdb) + ((1 - beta) * db)

        vdW = vdW / (1 - np.power(beta, t))
        vdb = vdb / (1 - np.power(beta, t))

        return (vdW, vdb)

    def compute_second_momentum(self, dW, db, sdW, sdb, beta, t):

        """
        Computes the second momentum for Adam optimization.

        Parameters:
            - dW (numpy.ndarray): Gradient of the loss with respect to weights.
            - db (numpy.ndarray): Gradient of the loss with respect to biases.
            - sdW (numpy.ndarray): Second moment estimate for weights.
            - sdb (numpy.ndarray): Second moment estimate for biases.
            - beta (float): Exponential decay rate for the second moment estimates.
            - t (int): Time step.

        Returns:
            - tuple: Tuple containing updated sdW and sdb.
        """

        sdW = (beta * sdW) + ((1 - beta) * np.power(dW, 2))
        sdb = (beta * sdb) + ((1 - beta) * np.power(db, 2))

        sdW = sdW / (1 - np.power(beta, t))
        sdb = sdb / (1 - np.power(beta, t))

        return (sdW, sdb)

    def update_parameters(self, dW, db, W, b, learning_rate, beta1, beta2, epsilon, t, adam_params=None):

        """
        Updates parameters using the Adam optimization algorithm.

        Parameters:
            - dW (numpy.ndarray): Gradient of the loss with respect to weights.
            - db (numpy.ndarray): Gradient of the loss with respect to biases.
            - W (numpy.ndarray): Weights of the layer.
            - b (numpy.ndarray): Biases of the layer.
            - learning_rate (float): Learning rate for the update.
            - beta1 (float): Exponential decay rate for the first moment estimates.
            - beta2 (float): Exponential decay rate for the second moment estimates.
            - epsilon (float): Small constant for numerical stability.
            - t (int): Time step.
            - adam_params (dict, optional): Dictionary containing existing Adam parameters (vdW, vdb, sdW, sdb).

        Returns:
            - tuple: Tuple containing updated W, b, and a dictionary of Adam parameters (vdW, vdb, sdW, sdb).
        """

        if adam_params is None:

            (vdW, vdb, sdW, sdb) = self.initialize_adam_parameters(W=W, b=b)

        else:

            vdW = adam_params["vdW"]
            vdb = adam_params["vdb"]
            sdW = adam_params["sdW"]
            sdb = adam_params["sdb"]

            (vdW, vdb) = self.compute_first_momentum(dW=dW, db=db, vdW=vdW, vdb=vdb, beta=beta1, t=t)
            (sdW, sdb) = self.compute_second_momentum(dW=dW, db=db, sdW=sdW, sdb=sdb, beta=beta2, t=t)

        W -= learning_rate * (vdW / np.sqrt(sdW + epsilon))
        b -= learning_rate * (vdb / np.sqrt(sdb + epsilon))

        adam_params_ = {

            "vdW": vdW,
            "vdb": vdb,
            "sdW": sdW,
            "sdb": sdb

        }

        return (W, b, adam_params_)


class MiniBatch:
    """
    This class implements functionality to create random mini-batches from the given dataset
    for stochastic gradient descent (SGD) optimization.

    Methods:
        - random_mini_batches(X, Y, mini_batch_size, seed):
            Creates a list of random mini-batches from the input data.
    """
    def __init__(self):
        """
        Initializes the MiniBatch class.
        """
        pass

    def random_mini_batches(self, X, Y, mini_batch_size, seed):

        """
        Creates a list of random mini-batches from the input data.

        Parameters:
            - X (numpy.ndarray): Input data of shape (m, n_x) where m is the number of examples
                                 and n_x is the number of features.
            - Y (numpy.ndarray): True labels of shape (n_y, m) where m is the number of examples
                                 and n_y is the number of output classes.
            - mini_batch_size (int): Size of each mini-batch.
            - seed (int): Random seed for reproducibility.

        Returns:
            - mini_batches (list): List of tuples (mini_batch_X, mini_batch_Y) where mini_batch_X has shape
                            (mini_batch_size, n_x) and mini_batch_Y has shape (n_y, mini_batch_size).
        """

        np.random.seed(seed)
        m = X.shape[0]

        permutation = np.random.permutation(m)
        shuffled_X = X[permutation]
        shuffled_Y = Y[:, permutation]

        num_complete_mini_batches = m // mini_batch_size
        mini_batches = []

        for k in range(num_complete_mini_batches):

            start_idx = k * mini_batch_size
            end_idx = (k + 1) * mini_batch_size
            mini_batch_X = shuffled_X[start_idx:end_idx]
            mini_batch_Y = shuffled_Y[:, start_idx:end_idx]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        if m % mini_batch_size != 0:

            start_idx = num_complete_mini_batches * mini_batch_size
            mini_batch_X = shuffled_X[start_idx:]
            mini_batch_Y = shuffled_Y[:, start_idx:]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        return mini_batches


class Losses:
    """
    This class implements different loss functions for training neural networks,
    including cross entropy and categorical cross entropy.

    Methods:
        - compute_cost(A, Y):
            Computes the cost given the predicted output and true labels.
    """

    def __init__(self, loss="cross entropy"):
        """
        Initializes the Losses class with the specified loss function.

        Parameters:
            - loss (str): Type of loss function to use ("cross entropy" or "categorical cross entropy").
                          Default is "cross entropy".
        """
        self.loss = loss

    def compute_cost(self, A, Y):
        """
        Computes the cost given the predicted output and true labels.

        Parameters:
            - A (numpy.ndarray): Predicted output from the model (probabilities).
            - Y (numpy.ndarray): True labels.

        Returns:
            - float: Computed cost value.

        Notes:
            This method is designed to be used with mini-batches, accumulating costs over an entire epoch
            and then dividing by the number of training examples.
        """
        cost = 0

        if self.loss == "cross entropy":
            log_probs = np.multiply(-np.log(A + 1e-15), Y) + np.multiply(-np.log(1 - A + 1e-15), 1 - Y)
            cost = np.sum(log_probs)

        elif self.loss == "categorical cross entropy":
            A = np.clip(A, 1e-15, 1 - 1e-15)
            cost = -np.sum(Y * np.log(A))

        return cost




class ConvolutionModel:
    """
    This class implements a Convolutional Neural Network (CNN) model training from scratch using a specific architecture.

    The architecture includes the following layers:
        1. Convolution
        2. Pooling
        3. Convolution
        4. Pooling
        5. Fully Connected
        6. Fully Connected

    Methods:
        - train(X, Y, epochs=10, mini_batch_size=64, alpha=1e-3):
            Trains the CNN model on the provided dataset.
    """
    def __init__(self):
        """
        Initializes the ConvolutionModel class.
        """
        pass

    def train(self, X, Y, epochs=10, mini_batch_size=64, alpha=1e-3):
        """
        Trains the CNN model on the provided dataset using the specified parameters.

        Parameters:
            - X (numpy.ndarray): Input data of shape (m, height, width, channels).
            - Y (numpy.ndarray): True labels of shape (m,).
            - epochs (int): Number of epochs to train the model. Default is 10.
            - mini_batch_size (int): Size of the mini-batches. Default is 64.
            - alpha (float): Learning rate for the optimizer. Default is 1e-3.

        Returns:
            - params (dict): Dictionary containing the trained parameters.
            - costs (list): List of costs calculated at each epoch.
            - train_time (float): Duration of the training process.

        Notes:
            The model follows a specific architecture with convolutional, pooling, and fully connected layers.
            It uses the Adam optimizer for parameter updates and the categorical cross-entropy loss function.
        """
        tic = time.time()
        costs = []

        num_classes = len(np.unique(Y))  # 6 classes
        Y = np.eye(num_classes)[Y].T
        m = X.shape[0]
        t = 1

        C1 = Conv2DForward()
        C2 = Conv2DBackward()
        P1 = Pool2DForward()
        P2 = Pool2DBackward()
        D1 = DenseFroward()
        D2 = DenseBackward()
        Adam = UpdateAdam()
        M = MiniBatch()
        L = Losses(loss="categorical cross entropy")

        print("---------------------------------------------------------------------------------")
        print()
        print("    My First Implementation of CNN From Scratch  (using loop!!! too slow :))     ")
        print()
        print("---------------------------------------------------------------------------------")

        #  Forward pass
        (A, conv_cache1, W1, b1) = C1.conv2d(X=X, filters=7, kernel_size=(3, 3), padding=1, strides=1, W=None, b=None)
        (vdW1, vdb1, sdW1, sdb1) = Adam.initialize_adam_parameters(W=W1, b=b1)
        (A, pool_cache1) = P1.pool_max_forward(A=A, f=2, stride=2)
        (A, conv_cache2, W2, b2) = C1.conv2d(X=A, filters=14, kernel_size=(3, 3), padding=1, strides=1, W=None, b=None)
        (vdW2, vdb2, sdW2, sdb2) = Adam.initialize_adam_parameters(W=W2, b=b2)
        (A, pool_cache2) = P1.pool_max_forward(A=A, f=2, stride=2)
        A_pool2 = A
        A = A.reshape(m, -1).T
        (A, W3, b3, fc_cache1) = D1.fc_forward(A=A, units=50, activation="relu", W=None, b=None, seed=0)
        (vdW3, vdb3, sdW3, sdb3) = Adam.initialize_adam_parameters(W=W3, b=b3)
        (A, W4, b4, fc_cache2) = D1.fc_forward(A=A, units=6, activation="softmax", W=None, b=None, seed=0)
        (vdW4, vdb4, sdW4, sdb4) = Adam.initialize_adam_parameters(W=W4, b=b4)
        cost = L.compute_cost(A, Y)
        costs.append(cost)
        print()
        print(f"                        Epoch 1;  Cost: {cost/m}                               ")

        # Backward pass
        (dA3, dW4, db4) = D2.fc_backward(A=A, Y=Y, cache=fc_cache2, activation="softmax", dA=None, last_layer=True)
        (vdW4, vdb4) = Adam.compute_first_momentum(dW=dW4, db=db4, vdW=vdW4, vdb=vdb4, beta=0.9, t=t)
        (sdW4, sdb4) = Adam.compute_second_momentum(dW=dW4, db=db4, sdW=sdW4, sdb=sdb4, beta=0.999, t=t)
        adam4 = {"vdW": vdW4, "vdb": vdb4, "sdW": sdW4, "sdb": sdb4}
        (W4, b4, adam4) = Adam.update_parameters(dW=dW4, db=db4, W=W4, b=b4, learning_rate=alpha, beta1=0.9, beta2=0.999,
                                            epsilon=1e-8, t=t, adam_params=adam4)
        (dA2, dW3, db3) = D2.fc_backward(A=fc_cache1[0][0], Y=Y, cache=fc_cache1, activation="relu", dA=dA3,
                                      last_layer=False)
        (vdW3, vdb3) = Adam.compute_first_momentum(dW=dW3, db=db3, vdW=vdW3, vdb=vdb3, beta=0.9, t=t)
        (sdW3, sdb3) = Adam.compute_second_momentum(dW=dW3, db=db3, sdW=sdW3, sdb=sdb3, beta=0.999, t=t)
        adam3 = {"vdW": vdW3, "vdb": vdb3, "sdW": sdW3, "sdb": sdb3}
        (W3, b3, adam3) = Adam.update_parameters(dW=dW3, db=db3, W=W3, b=b3, learning_rate=alpha, beta1=0.9, beta2=0.999,
                                            epsilon=1e-8, t=t, adam_params=adam3)
        dA2 = dA2.reshape(A_pool2.shape)
        dA2 = P2.pool_max_backward(dA=dA2, cache=pool_cache2)
        dA1, dW2, db2 = C2.conv2d_backward(dAl=dA2, caches=conv_cache2)
        (vdW2, vdb2) = Adam.compute_first_momentum(dW=dW2, db=db2, vdW=vdW2, vdb=vdb2, beta=0.9, t=t)
        (sdW2, sdb2) = Adam.compute_second_momentum(dW=dW2, db=db2, sdW=sdW2, sdb=sdb2, beta=0.999, t=t)
        adam2 = {"vdW": vdW2, "vdb": vdb2, "sdW": sdW2, "sdb": sdb2}
        (W2, b2, adam2) = Adam.update_parameters(dW=dW2, db=db2, W=W2, b=b2,
                                            learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-8, t=t,
                                            adam_params=adam2)
        dA1 = P2.pool_max_backward(dA=dA1, cache=pool_cache1)
        dA0, dW1, db1 = C2.conv2d_backward(dAl=dA1, caches=conv_cache1)
        (vdW1, vdb1) = Adam.compute_first_momentum(dW=dW1, db=db1, vdW=vdW1, vdb=vdb1, beta=0.9, t=t)
        (sdW1, sdb1) = Adam.compute_second_momentum(dW=dW1, db=db1, sdW=sdW1, sdb=sdb1, beta=0.999, t=t)
        adam1 = {"vdW": vdW1, "vdb": vdb1, "sdW": sdW1, "sdb": sdb1}
        (W1, b1, adam1) = Adam.update_parameters(dW=dW1, db=db1, W=W1, b=b1,
                                            learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-8, t=t,
                                            adam_params=adam1)

        for i in range(1, epochs):

            seed = 0
            mini_batches = M.random_mini_batches(X=X, Y=Y, mini_batch_size=mini_batch_size, seed=seed)
            total_cost = 0
            for j in range(len(mini_batches)):
                X_batch, Y_batch = mini_batches[j]
                m_ = X_batch.shape[0]

                # Forward pass
                (A_mini, conv_cache1_mini, W1, b1) = C1.conv2d(X=X_batch, filters=7, kernel_size=(3, 3), padding=1, strides=1, W=W1,
                                                  b=b1)
                (A_mini, pool_cache1_mini) = P1.pool_max_forward(A=A_mini, f=2, stride=2)
                (A_mini, conv_cache2_mini, W2, b2) = C1.conv2d(X=A_mini, filters=14, kernel_size=(3, 3), padding=1, strides=1, W=W2,
                                                  b=b2)
                (A_mini, pool_cache2_mini) = P1.pool_max_forward(A=A_mini, f=2, stride=2)
                A_pool2 = A_mini
                A_mini = A_mini.reshape(m_, -1).T
                (A_mini, W3, b3, fc_cache1_mini) = D1.fc_forward(A=A_mini, units=50, activation="relu", W=W3, b=b3, seed=0)
                (A_mini, W4, b4, fc_cache2_mini) = D1.fc_forward(A=A_mini, units=6, activation="softmax", W=W4, b=b4, seed=0)

                # compute cost
                cost = L.compute_cost(A_mini, Y_batch)
                costs.append(cost)
                total_cost += cost

                # backward pass
                (dA3, dW4, db4) = D2.fc_backward(A=A_mini, Y=Y_batch, cache=fc_cache2_mini, activation="softmax", dA=None,
                                              last_layer=True)
                (vdW4, vdb4) = Adam.compute_first_momentum(dW=dW4, db=db4, vdW=vdW4, vdb=vdb4, beta=0.9, t=t)
                (sdW4, sdb4) = Adam.compute_second_momentum(dW=dW4, db=db4, sdW=sdW4, sdb=sdb4, beta=0.999, t=t)
                adam4 = {"vdW": vdW4, "vdb": vdb4, "sdW": sdW4, "sdb": sdb4}
                (W4, b4, adam4) = Adam.update_parameters(dW=dW4, db=db4, W=W4, b=b4, learning_rate=alpha, beta1=0.9,
                                                    beta2=0.999, epsilon=1e-8, t=t, adam_params=adam4)

                (dA2, dW3, db3) = D2.fc_backward(A=fc_cache1_mini[0][0], Y=Y_batch, cache=fc_cache1_mini,
                                              activation="relu", dA=dA3, last_layer=False)
                (vdW3, vdb3) = Adam.compute_first_momentum(dW=dW3, db=db3, vdW=vdW3, vdb=vdb3, beta=0.9, t=t)
                (sdW3, sdb3) = Adam.compute_second_momentum(dW=dW3, db=db3, sdW=sdW3, sdb=sdb3, beta=0.999, t=t)
                adam3 = {"vdW": vdW3, "vdb": vdb3, "sdW": sdW3, "sdb": sdb3}
                (W3, b3, adam3) = Adam.update_parameters(dW=dW3, db=db3, W=W3, b=b3, learning_rate=alpha, beta1=0.9,
                                                    beta2=0.999, epsilon=1e-8, t=t, adam_params=adam3)

                dA2 = dA2.reshape(A_pool2.shape)
                dA2 = P2.pool_max_backward(dA=dA2, cache=pool_cache2)
                dA1, dW2, db2 = C2.conv2d_backward(dAl=dA2, caches=conv_cache2)
                (vdW2, vdb2) = Adam.compute_first_momentum(dW=dW2, db=db2, vdW=vdW2, vdb=vdb2, beta=0.9, t=t)
                (sdW2, sdb2) = Adam.compute_second_momentum(dW=dW2, db=db2, sdW=sdW2, sdb=sdb2, beta=0.99, t=t)
                adam2 = {"vdW": vdW2, "vdb": vdb2, "sdW": sdW2, "sdb": sdb2}
                (W2, b2, adam2) = Adam.update_parameters(dW=dW2, db=db2, W=W2, b=b2,
                                                    learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-8, t=t,
                                                    adam_params=adam2)

                dA1 = P2.pool_max_backward(dA=dA1, cache=pool_cache1)

                dA0, dW1, db1 = C2.conv2d_backward(dAl=dA1, caches=conv_cache1)
                (vdW1, vdb1) = Adam.compute_first_momentum(dW=dW1, db=db1, vdW=vdW1, vdb=vdb1, beta=0.9, t=t)
                (sdW1, sdb1) = Adam.compute_second_momentum(dW=dW1, db=db1, sdW=sdW1, sdb=sdb1, beta=0.999, t=t)
                adam1 = {"vdW": vdW1, "vdb": vdb1, "sdW": sdW1, "sdb": sdb1}
                (W1, b1, adam1) = Adam.update_parameters(dW=dW1, db=db1, W=W1, b=b1,
                                                    learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-8, t=t,
                                                    adam_params=adam1)
                t += 1
            seed += 1
            print()
            print(f"                       Epoch {i + 1};  Cost: {total_cost / m}                    ")

        params = {

            "W1": conv_cache1[0][1],
            "b1": conv_cache1[0][2],
            "W2": conv_cache2[0][1],
            "b2": conv_cache2[0][2],
            "W3": W3,
            "b3": b3,
            "W4": W4,
            "b4": b4
        }
        toc = time.time()

        train_time = toc - tic
        print()
        print(f"                        Train Duration: {train_time}")

        return (params, costs, train_time)


def load_dataset(train_path, test_path):
    train_data = h5py.File(train_path, "r")
    test_data = h5py.File(test_path, "r")
    X_train = np.array(train_data["train_set_x"])
    Y_train = np.array(train_data["train_set_y"])
    X_test = np.array(test_data["test_set_x"])
    Y_test = np.array(test_data["test_set_y"])
    classes = np.array(test_data["list_classes"])

    return X_train, Y_train, X_test, Y_test, classes


def preprocess_data(X):
    X = X / 255.
    X = np.transpose(X, (0, 1, 2, 3))

    return X

X_train, Y_train, X_test, Y_test, classes = load_dataset("/home/samani/Documents/projects/deep-learning/data/train_signs.h5", "/home/samani/Documents/projects/deep-learning/data/test_signs.h5")

X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

model = ConvolutionModel()

result = model.train(X_test[10:30, :, :, :], Y_test[10:30], epochs=10, mini_batch_size=5, alpha=1e-4)



"""
---------------------------------------------------------------------------------

    My First Implementation of CNN From Scratch  (using loop!!! too slow :))     

---------------------------------------------------------------------------------

                        Epoch 1;  Cost: 11.375739698309847                               

                        Epoch 2;  Cost: 9.479406466075112                    

                        Epoch 3;  Cost: 9.33823893318924                    

                        Epoch 4;  Cost: 9.338147874129124                    

                        Epoch 5;  Cost: 9.338147426441981                    

                        Epoch 6;  Cost: 9.338147412629644                    

                        Epoch 7;  Cost: 9.338147412136792                    

                        Epoch 8;  Cost: 9.338147412116745                    

                        Epoch 9;  Cost: 9.338147412115797                    

                        Epoch 10;  Cost: 9.338147412115745                    

                        Train Duration: 913.0007016658783

"""






