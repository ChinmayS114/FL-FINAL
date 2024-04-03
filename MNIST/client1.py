import flwr as fl
import tensorflow as tf
import wandb
import numpy as np

# Initialize wandb with your project and API key
wandb.init(project="MNIST", entity="chinmaysharmagsbf", config={"key": "c7f3b5bc64aed360788eab6b42588bf1a4018484"})

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

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

class MnistClient1(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        # Filter the data for digits 0-4
        x_train_client1 = []
        y_train_client1 = []
        for x, y in zip(x_train, y_train):
            if y < 5:
                x_train_client1.append(x)
                y_train_client1.append(y)
        x_train_client1 = np.array(x_train_client1)
        y_train_client1 = np.array(y_train_client1)
        history = model.fit(x_train_client1, y_train_client1, epochs=1, batch_size=32)
        wandb.log({"loss": history.history["loss"][0], "accuracy": history.history["accuracy"][0]})
        return model.get_weights(), len(x_train_client1), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # Filter the data for digits 0-4
        x_test_client1 = x_test[y_test < 5]
        y_test_client1 = y_test[y_test < 5]
        loss, accuracy = model.evaluate(x_test_client1, y_test_client1)
        wandb.log({"test_loss": loss, "test_accuracy": accuracy})
        return loss, len(x_test_client1), {"accuracy": accuracy}

# Start Flower client for Client 1
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MnistClient1())
