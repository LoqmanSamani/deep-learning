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



