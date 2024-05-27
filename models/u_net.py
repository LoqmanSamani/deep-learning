import tensorflow as tf
import numpy as np



class UNet:
    def __init__(
            self,
            classes,
            input_size=(96, 128, 3),
            filters=32,
            kernel_size=(3, 3),
            optimizer="adam",
            padding="same",
            strides=(2, 2),
            activation="relu",
            no_dropout=0,
            dropout=0.3,
            max_pooling=True,
            pool_size=(2, 2),
            conv_kernel_initializer="he_normal",
            u_conv_kernel_initializer="he_normal"
    ):

        self.input_size = input_size,
        self.classes = classes
        self.filters = filters,
        self.kernel_size = kernel_size,
        self.optimizer = optimizer
        self.padding = padding,
        self.strides = strides,
        self.activation = activation,
        self.no_dropout = no_dropout,
        self.dropout = dropout,
        self.max_pooling = max_pooling,
        self.pool_size = pool_size,
        self.conv_kernel_initializer = conv_kernel_initializer,
        self.u_conv_kernel_initializer = u_conv_kernel_initializer

    def convolution_block(
            self,
            inputs,
            filters,
            kernel_size,
            padding,
            activation,
            dropout_prob,
            kernel_initializer,
            max_pooling, pool_size
    ):

        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            kernel_initializer=kernel_initializer
        )(inputs)
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            kernel_initializer=kernel_initializer
        )(conv)

        if dropout_prob > 0:
            conv = tf.keras.layers.Dropout(rate=dropout_prob)(conv)

        if max_pooling:
            next_layer = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv)

        else:
            next_layer = conv

        skip_connection = conv

        return (next_layer, skip_connection)

    def up_convolution_block(
            self,
            inputs,
            skip_connection,
            filters,
            kernel_size,
            padding,
            strides,
            activation,
            kernel_initializer
    ):

        up_conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=activation,
            kernel_initializer=kernel_initializer
        )(inputs)

        merge = tf.keras.layers.Concatenate(axis=3)([up_conv, skip_connection])
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer
        )(merge)
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer
        )(conv)

        return conv


    def model(
            self,
            input_size,
            filters,
            kernel_size,
            classes,
            padding,
            strides,
            activation,
            no_dropout,
            dropout,
            max_pooling,
            pool_size,
            conv_kernel_initializer,
            u_conv_kernel_initializer
    ):

        input = tf.keras.Input(shape=input_size)

        conv_block1 = self.convolution_block(
            inputs=input,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            dropout_prob=no_dropout,
            kernel_initializer=conv_kernel_initializer,
            max_pooling=max_pooling,
            pool_size=pool_size
        )
        conv_block2 = self.convolution_block(
            inputs=conv_block1[0],
            filters=2*filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            dropout_prob=no_dropout,
            kernel_initializer=conv_kernel_initializer,
            max_pooling=max_pooling,
            pool_size=pool_size
        )
        conv_block3 = self.convolution_block(
            inputs=conv_block2[0],
            filters=4*filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            dropout_prob=no_dropout,
            kernel_initializer=conv_kernel_initializer,
            max_pooling=max_pooling,
            pool_size=pool_size
        )
        conv_block4 = self.convolution_block(
            inputs=conv_block3[0],
            filters=8*filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            dropout_prob=dropout,
            kernel_initializer=conv_kernel_initializer,
            max_pooling=max_pooling,
            pool_size=pool_size
        )
        conv_block5 = self.convolution_block(
            inputs=conv_block4[0],
            filters=16*filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            dropout_prob=dropout,
            kernel_initializer=conv_kernel_initializer,
            max_pooling=False,
            pool_size=pool_size
        )

        up_conv_block6 = self.up_convolution_block(
            inputs=conv_block5[0],
            skip_connection=conv_block4[1],
            filters=8*filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=activation,
            kernel_initializer=u_conv_kernel_initializer
        )
        up_conv_block7 = self.up_convolution_block(
            inputs=up_conv_block6,
            skip_connection=conv_block3[1],
            filters=4*filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=activation,
            kernel_initializer=u_conv_kernel_initializer
        )
        up_conv_block8 = self.up_convolution_block(
            inputs=up_conv_block7,
            skip_connection=conv_block2[1],
            filters=2*filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=activation,
            kernel_initializer=u_conv_kernel_initializer
        )
        up_conv_block9 = self.up_convolution_block(
            inputs=up_conv_block8,
            skip_connection=conv_block1[1],
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=activation,
            kernel_initializer=u_conv_kernel_initializer
        )

        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            kernel_initializer=u_conv_kernel_initializer
        )(up_conv_block9)

        conv1 = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            padding=padding
        )(conv)

        model = tf.keras.Model(
            inputs=input,
            outputs=conv1,
        )

        return model

    def train(
            self,
            train_dataset,
            epochs=10,
            learning_rate=1e-4,
            batch_size=16
    ):

        model = self.model(
            input_size=self.input_size,
            filters=self.filters,
            kernel_size=self.kernel_size,
            classes=self.classes,
            padding=self.padding,
            strides=self.strides,
            activation=self.activation,
            no_dropout=self.no_dropout,
            dropout=self.dropout,
            max_pooling=self.max_pooling,
            pool_size=self.pool_size,
            conv_kernel_initializer=self.conv_kernel_initializer,
            u_conv_kernel_initializer=self.u_conv_kernel_initializer
        )
        if self.optimizer == "adam":
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
        elif self.optimizer == "SGD":
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
        #  add more optimizers!

        model_history = model.fit(train_dataset, epochs=epochs, batch_size=batch_size)

        return model_history


model = UNet(input_size=(96, 128, 3), classes=20)
unet_model = model.model(
    input_size=(96, 128, 3),
    filters=32,
    kernel_size=(3, 3),
    classes=20,
    padding="same",
    strides=(2, 2),
    activation="relu",
    no_dropout=0,
    dropout=0.3,
    max_pooling=True,
    pool_size=(2, 2),
    conv_kernel_initializer="he_normal",
    u_conv_kernel_initializer="he_normal"
)
unet_model.summary()
"""
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 96, 128,   │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (Conv2D)     │ (None, 96, 128,   │        896 │ input_layer[0][0] │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_1 (Conv2D)   │ (None, 96, 128,   │      9,248 │ conv2d[0][0]      │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d       │ (None, 48, 64,    │          0 │ conv2d_1[0][0]    │
│ (MaxPooling2D)      │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_2 (Conv2D)   │ (None, 48, 64,    │     18,496 │ max_pooling2d[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_3 (Conv2D)   │ (None, 48, 64,    │     36,928 │ conv2d_2[0][0]    │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_1     │ (None, 24, 32,    │          0 │ conv2d_3[0][0]    │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_4 (Conv2D)   │ (None, 24, 32,    │     73,856 │ max_pooling2d_1[… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_5 (Conv2D)   │ (None, 24, 32,    │    147,584 │ conv2d_4[0][0]    │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (None, 12, 16,    │          0 │ conv2d_5[0][0]    │
│ (MaxPooling2D)      │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_6 (Conv2D)   │ (None, 12, 16,    │    295,168 │ max_pooling2d_2[… │
│                     │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_7 (Conv2D)   │ (None, 12, 16,    │    590,080 │ conv2d_6[0][0]    │
│                     │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout (Dropout)   │ (None, 12, 16,    │          0 │ conv2d_7[0][0]    │
│                     │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_3     │ (None, 6, 8, 256) │          0 │ dropout[0][0]     │
│ (MaxPooling2D)      │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_8 (Conv2D)   │ (None, 6, 8, 512) │  1,180,160 │ max_pooling2d_3[… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_9 (Conv2D)   │ (None, 6, 8, 512) │  2,359,808 │ conv2d_8[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_1 (Dropout) │ (None, 6, 8, 512) │          0 │ conv2d_9[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose    │ (None, 12, 16,    │  1,179,904 │ dropout_1[0][0]   │
│ (Conv2DTranspose)   │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (None, 12, 16,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 512)              │            │ dropout[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_10 (Conv2D)  │ (None, 12, 16,    │  1,179,904 │ concatenate[0][0] │
│                     │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_11 (Conv2D)  │ (None, 12, 16,    │    590,080 │ conv2d_10[0][0]   │
│                     │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose_1  │ (None, 24, 32,    │    295,040 │ conv2d_11[0][0]   │
│ (Conv2DTranspose)   │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_1       │ (None, 24, 32,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 256)              │            │ conv2d_5[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_12 (Conv2D)  │ (None, 24, 32,    │    295,040 │ concatenate_1[0]… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_13 (Conv2D)  │ (None, 24, 32,    │    147,584 │ conv2d_12[0][0]   │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose_2  │ (None, 48, 64,    │     73,792 │ conv2d_13[0][0]   │
│ (Conv2DTranspose)   │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_2       │ (None, 48, 64,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 128)              │            │ conv2d_3[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_14 (Conv2D)  │ (None, 48, 64,    │     73,792 │ concatenate_2[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_15 (Conv2D)  │ (None, 48, 64,    │     36,928 │ conv2d_14[0][0]   │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose_3  │ (None, 96, 128,   │     18,464 │ conv2d_15[0][0]   │
│ (Conv2DTranspose)   │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_3       │ (None, 96, 128,   │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 64)               │            │ conv2d_1[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_16 (Conv2D)  │ (None, 96, 128,   │     18,464 │ concatenate_3[0]… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_17 (Conv2D)  │ (None, 96, 128,   │      9,248 │ conv2d_16[0][0]   │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_18 (Conv2D)  │ (None, 96, 128,   │      9,248 │ conv2d_17[0][0]   │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_19 (Conv2D)  │ (None, 96, 128,   │        660 │ conv2d_18[0][0]   │
│                     │ 20)               │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 8,640,372 (32.96 MB)
 Trainable params: 8,640,372 (32.96 MB)
 Non-trainable params: 0 (0.00 B)
"""

















