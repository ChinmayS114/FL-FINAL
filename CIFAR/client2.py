import flwr as fl
import tensorflow as tf
import wandb
import numpy as np

# Initialize wandb with your project and API key
wandb.init(project="CIFAR-10", entity="chinmaysharmagsbf", config={"key": "c7f3b5bc64aed360788eab6b42588bf1a4018484"})

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        # Filter the data for a subset of classes
        classes_to_keep = [0, 1, 2, 3, 4]  # Define which classes to keep
        x_train_client = []
        y_train_client = []
        for x, y in zip(x_train, y_train):
            if y in classes_to_keep:
                x_train_client.append(x)
                y_train_client.append(y)
        x_train_client = np.array(x_train_client)
        y_train_client = np.array(y_train_client)
        history = model.fit(x_train_client, y_train_client, epochs=1, batch_size=32)
        wandb.log({"loss": history.history["loss"][0], "accuracy": history.history["accuracy"][0]})
        return model.get_weights(), len(x_train_client), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # Filter the data for a subset of classes
        classes_to_keep = [0, 1, 2, 3, 4]  # Define which classes to keep
        x_test_client = x_test[np.isin(y_test, classes_to_keep)]
        y_test_client = y_test[np.isin(y_test, classes_to_keep)]
        loss, accuracy = model.evaluate(x_test_client, y_test_client)
        wandb.log({"test_loss": loss, "test_accuracy": accuracy})
        return loss, len(x_test_client), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())
