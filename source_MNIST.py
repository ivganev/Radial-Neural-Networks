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

from torch.utils.data import (Dataset, TensorDataset, DataLoader)
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

def random_uniform_in_ball(m, d=28*28):
    '''
    Create random samples that are uniform inside the d-dimensional unit ball
    
    m is the number of samples,
    d is the total dimension, which is 28*28 for MNIST
    '''
    
    u = np.random.multivariate_normal(np.zeros(d),np.eye(d),m)  # an array of d normally distributed random variables
    norm=np.sum(u**2, axis = 1) **(0.5)
    norm = norm.reshape(m,1)
    rands = np.random.uniform(size=m)**(1.0/d)
    rands = rands.reshape(m,1)
    return rands*u/norm

# Note: need to do the following before adding to a sample:
# torch.tensor(random_uniform_in_ball(m,radius,d)).reshape(m,1,28,28)

# Calculate distance from each sample to the nearest other one
def shortest_distances(x: torch.tensor) -> list:
    result = []
    for i in range(len(x)):
        radius = float('inf')
        for j in range(i):
            if torch.linalg.norm(x[i] - x[j]).item() < radius:
                radius = torch.linalg.norm(x[i] - x[j]).item()
        for j in range(i+1,len(x)):
            if torch.linalg.norm(x[i] - x[j]).item() < radius:
                radius = torch.linalg.norm(x[i] - x[j]).item()
        result.append(radius)
    return result

def num_classes(dataset):
    '''
    The number of classes in a classification dataset
    '''
    x,y = next(iter(dataset))
    return y.shape[-1]

####################################
####### Get MNIST data
####################################

mnist_training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

####################################
####### Make selection and add noise
####################################

def noisy_mnist_dataset(label, n=5, m=100, verbose=False, noise_scale=0.5):
    '''
    Generate a dataset with multiple noisy copies of MNIST images
    
    Parameters:
      label is one of 0,1,2,3,4,5,6,7,8,9;
      n is the number of original images; 
      m is the number of noisy samples per original image;
      noise_scale is the radius of the ball
    
    Returns: a torch Dataset of size n*m, where
      x is a noisy image
      y is the identity of that image
    '''
    
    # Get enough MNIST samples to work with
    batch_size = n * 100
    train_dataloader = DataLoader(mnist_training_data, batch_size, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    
    # Get a selection of data with the same label, crop it to size n
    selection = train_features[train_labels == label]
    selection = selection[:n]
    assert selection.shape[0] == n, "Too few of this label; take another batch"
    if verbose:
        plt.imshow(selection[2].squeeze(), cmap="gray")
        plt.show()
    
    radii = shortest_distances(selection)
    noise = torch.tensor(random_uniform_in_ball(m, d=28*28)).reshape(m,1,28,28)
    
    noisy_samples = torch.Tensor(torch.Size([int(n*m), 1, 28, 28]))
    noisy_labels = torch.Tensor(torch.Size([n*m])).to(torch.int64)
    for i in range(n):
        radius = radii[i] * noise_scale
        assert radius > 1e-6, "some samples are too close together"
        for j in range(m):
            noisy_samples[i*m + j] = selection[i] + radius*noise[j]   
            noisy_labels[i*m + j] = i
    
    if verbose:
        plt.imshow(selection[0][0], cmap="gray")
        plt.show()
        plt.imshow(noisy_samples[0][0], cmap="gray")
        plt.show()
        plt.imshow(noisy_samples[1][0], cmap="gray")
    
    return TensorDataset(noisy_samples.flatten(1), noisy_labels)

# Alternative for the labels:
# torch.eye(n).repeat_interleave(m, dim=0)

####################################
####### MLP network equivalent to RadNet
####################################

def make_relu_network(dims):
    '''
    Build a ReLU network with the given layer sizes.
    There is a ReLU between each pair of linear layers.
    '''
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i+1]))
        if i+1 < len(dims):
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

####################################
####### Training and testing
####################################

def test(model, test_data, loss_fn=torch.nn.CrossEntropyLoss(), device=device):
    '''
    Evaluate a model on a test set, return loss and accuracy
    '''
    with torch.no_grad():
        loss_sum = 0.0
        accuracy_sum = 0.0
        sample_count = 0
        for x, y in test_data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            pred_class = torch.argmax(y_pred, dim=1)
            accuracy = torch.mean((pred_class == y).to(float))
            loss_sum += loss.detach().cpu().item() * len(y)
            accuracy_sum += accuracy.detach().cpu().item() * len(y)
            sample_count += len(y)
        return loss_sum / sample_count, accuracy_sum / sample_count

def training_loop_with_test(epochs, learning_rate, model, train_data, test_data=None, verbose=False, test_interval=500, loss_fn=torch.nn.CrossEntropyLoss(), device=device):
    log = {
      'train_loss': [],
      'train_accuracy': [],
      'test_epoch': [],
      'test_loss': [],
      'test_accuracy': [],
    }
    model.train(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        accuracy_sum = 0.0
        sample_count = 0
        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x) 
            loss = loss_fn(y_pred, y)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred_class = torch.argmax(y_pred, dim=1)
                accuracy = torch.mean((pred_class == y).to(float))
                loss_sum += loss.detach().cpu().item() * len(y)
                accuracy_sum += accuracy.detach().cpu().item() * len(y)
                sample_count += len(y)
        
        train_loss = loss_sum / sample_count
        train_accuracy = accuracy_sum / sample_count
        log['train_loss'].append(train_loss)
        log['train_accuracy'].append(train_accuracy)
        
        if test_data is not None:
            if epoch == 1 or epoch % test_interval == 0 or epoch == epochs:
                test_loss, test_accuracy = test(model, test_data, loss_fn=loss_fn, device=device)
                log['test_epoch'].append(epoch)
                log['test_loss'].append(test_loss)
                log['test_accuracy'].append(test_accuracy)
        
        if verbose:
            if epoch == 1 or epoch % test_interval == 0 or epoch == epochs:
                if test_data is not None:
                    print('Epoch %d, Train loss %f, Train accuracy %f, Test loss %f, Test accuracy %f'
                           % (epoch, train_loss, train_accuracy, test_loss, test_accuracy))
                else:
                    print('Epoch %d, Train loss %f, Train accuracy %f'
                           % (epoch, train_loss, train_accuracy))

    model.train(False)
    return log

####################################
####### Train both a radnet and an MLP
####################################

def train_both(num_samples, m_copies, dim_vector, label=3, noise_scale=0.5, verbose=False, epochs=1000, lr_radnet = 0.05, lr_mlp=0.05, batch_size=None, loss_fn=torch.nn.CrossEntropyLoss(), train_fraction=0.8, test_interval=500, device=device):
    
    # Dataset
    data = noisy_mnist_dataset(label=label, n=int(num_samples), m=int(m_copies), verbose=False)
    # Split into train and test
    num_train = int(len(data) * train_fraction)
    train_data, test_data = torch.utils.data.random_split(data, [num_train, len(data)- num_train])
    # Construct DataLoaders with given batch size, or full batch
    if batch_size is None:
        batch_size = len(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=len(test_data))
    
    if verbose:
        print('')
        print('### Data description')
        print('number or original images =', num_samples)
        print('number of copies of each =', m_copies)
        print('number or samples =', len(train_data), "train,", len(test_data), "test")
        print('dimension vector =', dim_vector)
        print('device =', device)
        print('')
    
    if verbose:
        print('#### Training stepReLU radnet:')
    radnet = RadNet(eta=stepReLU_eta, dims=dim_vector, has_bias=False).to(device)
    radnet_log = training_loop_with_test(
        epochs = epochs, 
        learning_rate = lr_radnet,
        model = radnet,
        train_data = train_loader,
        test_data = test_loader,
        loss_fn = loss_fn,
        verbose = verbose,
        test_interval = test_interval,
        device = device)
    
    if verbose:  
        print('')    
        print('#### Training ReLU MLP:')
    relu_net = make_relu_network(dim_vector).to(device)
    relu_log = training_loop_with_test(
        epochs = epochs, 
        learning_rate = lr_mlp,
        model = relu_net,
        train_data = train_loader,
        test_data = test_loader,
        loss_fn = loss_fn,
        verbose = verbose,
        test_interval = test_interval,
        device = device)
    
    return {'RadNet': radnet_log, 'ReLU MLP': relu_log}

