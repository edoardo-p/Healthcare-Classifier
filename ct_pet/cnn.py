import tensorflow as tf


class CTConv(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv3D(32, 16, activation="relu")
        self.conv2 = tf.keras.layers.Conv3D(64, 16, activation="relu")
        self.conv3 = tf.keras.layers.Conv3D(128, 16, activation="relu")
        self.pooling = tf.keras.layers.MaxPool3D()

        self.flatten = tf.keras.layers.Flatten()
        # self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.pooling(x)

        x = self.flatten(x)
        # x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


if __name__ == "__main__":
    model = CTConv()
    model.build((None, 128, 512, 512, 1))
    model.summary()
