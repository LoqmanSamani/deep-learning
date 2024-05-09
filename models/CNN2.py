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


