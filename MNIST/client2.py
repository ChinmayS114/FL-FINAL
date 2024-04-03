import flwr as fl
import tensorflow as tf
import numpy as np

# Load model and data (MobileNetV2, MNIST)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class MnistClient2(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        # Load MNIST data
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
        y_train = y_train.astype("int32")
        x_train_client2 = []
        y_train_client2 = []
        for x, y in zip(x_train, y_train):
            if y >= 5:
                x_train_client2.append(x)
                y_train_client2.append(y)
        x_train_client2 = np.array(x_train_client2)
        y_train_client2 = np.array(y_train_client2)
        x_train_client2_noisy = self.add_noise(x_train_client2)
        model.fit(x_train_client2_noisy, y_train_client2, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train_client2_noisy), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # Load MNIST data
        _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
        y_test = y_test.astype("int32")
        # Filter the data for digits 5-9
        x_test_client2 = x_test[y_test >= 5]
        y_test_client2 = y_test[y_test >= 5]
        loss, accuracy = model.evaluate(x_test_client2, y_test_client2)
        return loss, len(x_test_client2), {"accuracy": accuracy}
    
    def add_noise(self, data):
        noise_factor = 0.2
        noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return np.clip(noisy_data, 0.0, 1.0)

# Start Flower client for Client 2
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MnistClient2())
