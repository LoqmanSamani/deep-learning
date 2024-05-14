import numpy as np
import random


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

    def conv2d(self, X, filters, kernel_size=(3, 3), padding=2, strides=1, parameters=None, layer=None):
        """
        Applies 2D convolution operation to the input data.

        Args:
            - X (ndarray): Input data with shape (m, n_h, n_W, n_C).
            - filters (int): Number of applied filters.
            - kernel_size (tuple): Size of the applied filters (f, f).
            - padding (int): Number of padding columns or rows.
            - strides (int): Number of strides.
            - parameters (dict): Contains pre-initialized weight matrices W and bias vectors b.
            - layer (int): Layer number.

        Returns:
            tuple: A tuple containing the output data, cache, weight matrix W, and bias vector b.
        """

        num_channels = X[0].shape[-1]
        f = kernel_size[0]

        if not parameters:
            (W, b) = self.initialize_parameters(f=f, num_channels=num_channels, filters=filters)
        else:
            (W, b) = parameters[f"W{layer}"], parameters[f"b{layer}"]

        linear_cache = (X, W, b, filters, kernel_size, padding, strides, layer)
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
        (X, W, b, filters, kernel_size, padding, strides, layer) = linear_cache

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

        if not W and not b:
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



