import numpy as np
from typing import List

import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

import argparse
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from statistics import mean, stdev

from source import *
from MNIST_source import *


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

batch_size = 128
train_dataloader = DataLoader(training_data, batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
train_features_flat = train_features.flatten(1)
train_labels_onehot = F.one_hot(train_labels, num_classes=10)


radnet = RadNet(eta=torch.sigmoid, dims=[28*28,28*28, 28*28 , 28*28,10], has_bias=False)


model_trained, model_losses, model_accs = ce_training_loop(
    n_epochs = 3000, 
    learning_rate = 0.05,
    model = radnet,
    params = list(radnet.parameters()),
    x_train = train_features_flat,
    y_train = train_labels_onehot,
    verbose=True)

relu_net = torch.nn.Sequential(
    torch.nn.Linear(28*28, 28*28),
    torch.nn.ReLU(),
    torch.nn.Linear(28*28, 28*28),
    torch.nn.ReLU(),
    torch.nn.Linear(28*28, 28*28),
    torch.nn.ReLU(),
    torch.nn.Linear(28*28, 10)
    )

relu_model_trained, relu_model_losses, relu_model_accs = ce_training_loop(
    n_epochs = 3000, 
    learning_rate = 0.05,
    model = relu_net,
    params = list(relu_net.parameters()),
    x_train = train_features_flat,
    y_train = train_labels_onehot,
    verbose=True)

