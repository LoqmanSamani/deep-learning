import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import numpy as np


class MulticlassClassifier:
    def __init__(
            self,
            X_train,
            Y_train,
            layers,
            filters,
            pooling,
            units,
            num_classes,
            X_validation=None,
            Y_validation=None,
            X_test=None,
            Y_test=None,
            pooling_="max",
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            output_activation="sigmoid",
            optimizer="adam"
    ):

        self.X_train = X_train
        self.Y_train = Y_train
        self.layers = layers
        self.filters = filters
        self.pooling = pooling
        self.units = units
        self.num_classes = num_classes
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.X_test = X_test
        self.Y_test = Y_test
        self.pooling_ = pooling_
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_activation = output_activation
        self.optimizer = optimizer

    def build_layers(self, layers, filters, pooling, pooling_, units):

        layers_ = {}

        for i in range(1, len(layers)+1):

            if layers[i-1] == "conv":
                layers_[f"layer{i}"] = tf.keras.layers.Conv2D(filters=filters[i-1][0], kernel_size=filters[i-1][1], strides=filters[i-1][2], padding=filters[i-1][3])
            elif layers[i-1] == "pooling":
                if pooling_ == "average":
                    layers_[f"layer{i}"] = tf.keras.layers.AvgPool2D(pool_size=pooling[i-1][0], strides=pooling[i-1][1], padding=pooling[i-1][2])
                else:
                    layers_[f"layer{i}"] = tf.keras.layers.MaxPool2D(pool_size=pooling[i-1][0], strides=pooling[i-1][1], padding=pooling[i-1][2])
            elif layers[i-1] == "batch_norm":
                layers_[f"layer{i}"] = tf.keras.layers.BatchNormalization(axis=3)
            elif layers[i-1] == "fc":
                layers_[f"layer{i}"] = tf.keras.layers.Dense(units=units[i-1][0], activation=units[i-1][1])
            elif layers[i-1] == "relu":
                layers_[f"layer{i}"] = tf.keras.layers.ReLU()
            elif layers[i-1] == "flatten":
                layers_[f"layer{i}"] = tf.keras.layers.Flatten()

            else:
                raise ValueError("Please choose a proper layer!!!")

        return layers_

    def build_model(self, input_shape, num_layers, layers_):

        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = input_layer

        for i in range(1, num_layers + 1):
            x = layers_[f"layer{i}"](x)

        model = tf.keras.Model(inputs=input_layer, outputs=x)

        return model

    def one_hot_encoding(self, num_classes):

        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, num_classes=num_classes)
        if self.Y_validation:
            self.Y_validation = tf.keras.utils.to_categorical(self.Y_validation, num_classes=num_classes)
        if self.Y_test:
            self.Y_test = tf.keras.Units.to_categorical(self.Y_test, num_classes=num_classes)

    def train(self):

        input_shape = self.X_train[0].shape
        num_layers = len(self.layers)

        val_data = None
        test_data = None
        history = None

        if self.num_classes > 2:
            self.one_hot_encoding(num_classes=self.num_classes)
            loss = "categorical_crossentropy"
        else:
            loss = "binary_crossentropy"

        layers_ = self.build_layers(layers=self.layers, filters=self.filters, pooling=self.pooling, pooling_=self.pooling_, units=self.units)

        model = self.build_model(input_shape=input_shape, num_layers=num_layers, layers_=layers_)
        model.summary()

        if self.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        elif self.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


        train_data = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train)).batch(self.batch_size)

        if self.X_validation is not None and self.Y_validation is not None:
            val_data = tf.data.Dataset.from_tensor_slices((self.X_validation, self.Y_validation)).batch(self.batch_size)

        if self.X_test is not None and self.Y_test is not None:
            test_data = tf.data.Dataset.from_tensor_slices((self.X_test, self.Y_test)).batch(self.batch_size)

        if val_data:
            history = model.fit(train_data, epochs=self.epochs, validation_data=val_data)
        if test_data:
            model.evaluate(self.X_test, self.Y_test)
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["loss"], label="train loss")
        if val_data:
            plt.plot(history.history["val_loss"], label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("epoch vs. loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(history.history["accuracy"], label="train accuracy")
        if val_data:
            plt.plot(history.history["val_accuracy"], label="validation accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("epoch vs. accuracy")
        plt.legend()
        plt.show()

        return history



train_data = h5py.File("/home/samani/Documents/projects/deep-learning/data/train_happy.h5", "r")
test_data = h5py.File("/home/samani/Documents/projects/deep-learning/data/test_happy.h5", "r")
X_train = train_data["train_set_x"]
X_test = test_data["test_set_x"]
Y_train = train_data["train_set_y"]
Y_test = test_data["test_set_y"]

X_train = np.array(list(X_train))
X_test = np.array(list(X_test))

X_train = X_train / 255.
X_test = X_test / 255.


_layers = ("conv", "pooling", "batch_norm", "relu", "conv", "pooling", "batch_norm", "relu", "flatten", "fc")
_filters = ((7, (3, 3), (1, 1), "same"), "_", "_", "_", (10, (5, 5), (2, 2), "same"), "_", "_", "_", "_", "_")
_pooling = ("_", ((3, 3), (3, 3), "same"), "_", "_", "_", ((2, 2), (2, 2), "same"), "_", "_", "_", "_")
_units = ("_", "_", "_", "_", "_", "_", "_", "_", "_", (1, "sigmoid"))


model = MulticlassClassifier(
    X_train=X_train,
    Y_train=Y_train,
    layers=_layers,
    filters=_filters,
    pooling=_pooling,
    pooling_="max",
    units=_units,
    num_classes=2,
    X_validation=X_test,
    Y_validation=Y_test,
    X_test=None,
    Y_test=None,
    epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    output_activation="sigmoid",
    optimizer="adam"
)

model.train()

"""
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 64, 64, 3)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 64, 64, 7)      │           196 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 22, 22, 7)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 22, 22, 7)      │            28 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu (ReLU)                    │ (None, 22, 22, 7)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 11, 11, 10)     │         1,760 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 6, 6, 10)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 6, 6, 10)       │            40 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_1 (ReLU)                  │ (None, 6, 6, 10)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 360)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 1)              │           361 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,385 (9.32 KB)
 Trainable params: 2,351 (9.18 KB)
 Non-trainable params: 34 (136.00 B)
Epoch 1/50
38/38 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.4669 - loss: 0.7934 - val_accuracy: 0.5133 - val_loss: 0.6865
Epoch 2/50
38/38 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.6341 - loss: 0.6624 - val_accuracy: 0.4667 - val_loss: 0.6909
Epoch 3/50
38/38 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6871 - loss: 0.6218 - val_accuracy: 0.4733 - val_loss: 0.6910

   ...
   
Epoch 49/50
38/38 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.9794 - loss: 0.1234 - val_accuracy: 0.9533 - val_loss: 0.1903
Epoch 50/50
38/38 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.9794 - loss: 0.1204 - val_accuracy: 0.9533 - val_loss: 0.1880
"""


train_data = h5py.File("/home/samani/Documents/projects/deep-learning/data/train_signs.h5", "r")
test_data = h5py.File("/home/samani/Documents/projects/deep-learning/data/test_signs.h5", "r")
X_train = train_data["train_set_x"]
X_test = test_data["test_set_x"]
Y_train = train_data["train_set_y"]
Y_test = test_data["test_set_y"]

X_train = np.array(list(X_train))
X_test = np.array(list(X_test))

X_train = X_train / 255.
X_test = X_test / 255.


_layers = ("conv", "pooling", "batch_norm", "relu", "conv", "pooling", "batch_norm", "relu", "flatten", "fc", "fc")
_filters = ((7, (3, 3), (1, 1), "same"), "_", "_", "_", (10, (5, 5), (2, 2), "same"), "_", "_", "_", "_", "_", "_")
_pooling = ("_", ((3, 3), (3, 3), "same"), "_", "_", "_", ((2, 2), (2, 2), "same"), "_", "_", "_", "_", "_")
_units = ("_", "_", "_", "_", "_", "_", "_", "_", "_", (20, "relu"), (6, "softmax"))


model = MulticlassClassifier(
    X_train=X_train,
    Y_train=Y_train,
    layers=_layers,
    filters=_filters,
    pooling=_pooling,
    pooling_="max",
    units=_units,
    num_classes=6,
    X_validation=X_test,
    Y_validation=Y_test,
    X_test=None,
    Y_test=None,
    epochs=70,
    batch_size=32,
    learning_rate=1e-4,
    output_activation="softmax",
    optimizer="adam"
)

model.train()

"""
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 64, 64, 3)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 64, 64, 7)      │           196 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 22, 22, 7)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 22, 22, 7)      │            28 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu (ReLU)                    │ (None, 22, 22, 7)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 11, 11, 10)     │         1,760 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 6, 6, 10)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 6, 6, 10)       │            40 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_1 (ReLU)                  │ (None, 6, 6, 10)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 360)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 20)             │         7,220 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 6)              │           126 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 9,370 (36.60 KB)
 Trainable params: 9,336 (36.47 KB)
 Non-trainable params: 34 (136.00 B)

Epoch 1/70
34/34 ━━━━━━━━━━━━━━━━━━━━ 2s 20ms/step - accuracy: 0.1895 - loss: 1.9179 - val_accuracy: 0.1667 - val_loss: 1.8310
Epoch 2/70
34/34 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.2477 - loss: 1.7623 - val_accuracy: 0.1667 - val_loss: 1.8095
Epoch 3/70
34/34 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.3184 - loss: 1.6812 - val_accuracy: 0.1667 - val_loss: 1.7889

   ...
   
Epoch 70/70
34/34 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - accuracy: 0.9913 - loss: 0.1272 - val_accuracy: 0.9083 - val_loss: 0.2636
"""




