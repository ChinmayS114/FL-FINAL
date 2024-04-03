import flwr as fl
import wandb

# Initialize wandb
wandb.init(project="flower_server")

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)
