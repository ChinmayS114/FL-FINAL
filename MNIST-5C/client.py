import flwr as fl
import tensorflow as tf
import wandb
import threading

# Initialize wandb with your project and API key
wandb.init(project="MNIST", entity="chinmaysharmagsbf", config={"key": "c7f3b5bc64aed360788eab6b42588bf1a4018484"})

# Load model and data (MobileNetV2, MNIST)
def get_model():
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
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, idx):
        self.model = get_model()
        self.idx = idx

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(x_train, y_train, epochs=1, batch_size=32)
        wandb.log({f"Client {self.idx + 1} loss": history.history["loss"][0], f"Client {self.idx + 1} accuracy": history.history["accuracy"][0]})
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        wandb.log({f"Client {self.idx + 1} test_loss": loss, f"Client {self.idx + 1} test_accuracy": accuracy})
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower clients
def start_client(client_idx):
    client = MnistClient(client_idx)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

threads = []
for i in range(6):
    thread = threading.Thread(target=start_client, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
