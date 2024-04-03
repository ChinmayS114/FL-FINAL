import flwr as fl
import wandb

# Initialize wandb with your project and API key
wandb.init(project="MNIST", entity="chinmaysharmagsbf", config={"key": "c7f3b5bc64aed360788eab6b42588bf1a4018484"})

# Define Flower server configuration
server_config = fl.server.ServerConfig(num_rounds=3)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=server_config,
)
