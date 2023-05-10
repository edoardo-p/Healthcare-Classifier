import tensorflow as tf


class Conv2D(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv3D(32, 3, activation="relu")
        self.conv2 = tf.keras.layers.Conv3D(64, 3, activation="relu")
        self.conv3 = tf.keras.layers.Conv3D(128, 3, activation="relu")
        self.pooling = tf.keras.layers.MaxPool3D()

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, x):
        w, h, _ = x.shape
        if w != h or w not in (128, 256, 512):
            raise ValueError(
                "Input tensor must be square of sizes (128x128), (256x256) or (512x512)"
            )

        pooling = (w // 128, w // 128, 1)
        x = tf.keras.layers.MaxPool3D(pooling, strides=pooling)(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
