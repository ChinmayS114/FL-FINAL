Federated Learning with Flower: MNIST and CIFAR-10
This project demonstrates Federated Learning (FL) implementation using Flower framework with the MNIST and CIFAR-10 datasets. Flower enables collaborative training of machine learning models across decentralized devices.

Overview
In this project, we use Flower to implement Federated Learning with two popular datasets: MNIST (handwritten digit recognition) and CIFAR-10 (object recognition). The goal is to train deep learning models in a distributed manner, where training data remains on client devices, ensuring privacy and reducing communication overhead.

Features
Implementation of Federated Learning with Flower framework
Use of MNIST and CIFAR-10 datasets for training
Training and evaluation scripts for FL scenarios
Monitoring training progress with Flower's dashboard

Requirements
Python 3.6+
Flower
TensorFlow
WandB(for analytics)

Usage
Start the Flower server by running flower_server.py.

Launch multiple client devices by running client.py. Use --dataset flag to specify the dataset (mnist or cifar10).

Monitor training progress and model performance using Flower's dashboard (typically accessible at http://localhost:8080).

Experiment with different configurations (e.g., number of clients, learning rate, batch size) to observe their impact on FL outcomes.

Files Structure
flower_server.py: Flower server setup and configuration.
client.py: Client device script for FL training.
mnist_loader.py: Data loader and preprocessing functions for MNIST dataset.
cifar10_loader.py: Data loader and preprocessing functions for CIFAR-10 dataset.
models.py: Definition of neural network models (e.g., CNN) used for training.
requirements.txt: List of Python dependencies.
References
Flower Documentation: https://flower.dev/docs/
MNIST Dataset: http://yann.lecun.com/exdb/mnist/
CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
