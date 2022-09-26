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

from source import *


####################################
####### Set-up functions
####################################

# Step RelU
def stepReLU_eta(r):
    '''Version for the eta in the definition of a radial rescaling activation;
    r is the norm'''

    if r.shape == torch.Size([]):
        if r < 1:
            return 1e-6
        return r
    else:
        for i in range(len(r)):
            if r[i] < 1:
                r[i] = 1e-6
    return r

# Create uniform random noise in the unit d-ball
def generate_noise(m, rad, d=28*28):
    '''m is the number of samples, rad is the radius;
    d is the total dimension, which is 28*28 for MNIST'''
    
    u = np.random.multivariate_normal(np.zeros(d),np.eye(d),m)  # an array of d normally distributed random variables
    norm=np.sum(u**2, axis = 1) **(0.5)
    norm = norm.reshape(m,1)
    rands = np.random.uniform(size=m)**(1.0/d)
    rands = rands.reshape(m,1)
    return rad*rands*u/norm

# Calculate the smallest distance between vectors in a tensor
def smallest_distance(x: torch.tensor) -> float:
    min_dist = float('inf')
    for i in range(len(x)):
        for j in range(i):
            if torch.linalg.norm(x[i] - x[j]).item() < min_dist:
                min_dist = torch.linalg.norm(x[i] - x[j]).item()
    return min_dist

####################################
####### Get MNIST data
####################################

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

batch_size = 128
train_dataloader = DataLoader(training_data, batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

if False:
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


####################################
####### Make selection and add noise
####################################

## Select the threes
def add_noise(label, n=5, m=100, verbose=False):
    '''label is one of 0,1,2,3,4,5,6,7,8,9;
    n is the number of original images; 
    m is the number of noisy samples per original image;
    may want to make the 2.5 definition of the radius into a hyperparameter
    '''
    
    # Get a selection of data with the same label, crop it to size n
    selection = train_features[train_labels == label]
    selection = selection[:n]
    assert selection.shape[0] == n, "Too few of this label; take another batch"
    if verbose:
        plt.imshow(selection[2].squeeze(), cmap="gray")
        plt.show()
    
    radius = smallest_distance(selection)/2.5
    noise = torch.tensor(generate_noise(m,radius,d=28*28)).reshape(m,1,28,28)
    
    noisy_samples = torch.Tensor(torch.Size([int(n*m), 1, 28, 28]))
    noisy_labels = torch.Tensor(torch.Size([n*m, n]))
    for i in range(n):
        for j in range(m):
            noisy_samples[i*n + j]= selection[i] + noise[j]   
            noisy_labels[i*m + j]=  torch.eye(n)[i]
    
    if verbose:
        plt.imshow(selection[0][0], cmap="gray")
        plt.show()
        plt.imshow(noisy_samples[0][0], cmap="gray")
        plt.show()
        plt.imshow(noisy_samples[1][0], cmap="gray")
    
    return noisy_samples, noisy_labels

# Alternative for the labels:
# torch.eye(n).repeat_interleave(m, dim=0)

####################################
####### Train two networks
####################################

def train_both(num_samples, m_copies, dim_vector, label=3):
    noisy_threes, noisy_labels = add_noise(label=3, n=int(num_samples), m=int(m_copies), verbose =False)
    noisy_threes_flat = noisy_threes.flatten(1)
    
    print('')
    print('### Data description')
    print('number or original images =', num_samples)
    print('number of copies of each =', m_copies)
    print('dimension vector =', dim_vector)
    print('')
    
    print('#### Training stepReLU radnet:')
    radnet = RadNet(eta=stepReLU_eta, dims=dim_vector, has_bias=False)
    model_trained, model_losses = training_loop(
        n_epochs = 2000, 
        learning_rate = 0.05,
        model = radnet,
        params = list(radnet.parameters()),
        x_train = noisy_threes_flat,
        y_train = noisy_labels,
        verbose=True)
    
    relu_net = torch.nn.Sequential(
        torch.nn.Linear(28*28, dim_vector[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_vector[1], dim_vector[2]),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_vector[2], dim_vector[3]),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_vector[3],1)
        )
    
    print('')
    print('#### Training ReLU MLP:')
    
    relu_model_trained, relu_model_losses = training_loop(
        n_epochs = 2000, 
        learning_rate = 0.05,
        model = relu_net,
        params = list(relu_net.parameters()),
        x_train = noisy_threes_flat,
        y_train = noisy_labels,
        verbose=True)
    
    return
    

####################################
####### Run experiment
####################################

ns = [4,7,10]
ms = [100,500,1000]
d= 28*28
dim_vecs = [
    [d, d+1, d+2, d+3, 1],
    [d, d+1, d+2, d+3, d+4, 1],
    [d, d+1, d+2, d+3, d+4, d+5, 1]]

if False:
    for dims in dim_vecs:
        for n in ns:
            for m in ms:
                train_both(
                    num_samples = n,
                    m_copies = m,
                    dim_vector= dims)
                