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
from source_MNIST import *

#torch.manual_seed(1)
#np.random.seed(1)


def main():
    # Number of original images
    n = 3
    
    # Number of noisy samples for each original sample
    m = 100
    
    # Widths of both neural networks
    widths = [28*28, 28*28 + 1, 28*28 + 2, n]

    # Number of trials
    num_trials = 5
    
    # Number of epochs
    num_epochs = 150
    
    # Noise level
    noise_scale = 3

    for trial in tqdm(range(num_trials)):
        logs = train_both(
                num_samples = n,
                m_copies = m,
                dim_vector = widths,
                verbose=False,
                device='cpu',
                epochs=num_epochs,
                lr_radnet = 0.05, 
                lr_mlp = 0.05,
                noise_scale = noise_scale)

        rad_loss  = logs['RadNet']['train_loss']
        relu_loss = logs['ReLU MLP']['train_loss']
        if trial == 0:
            plt.plot(torch.tensor(rad_loss).detach(), color='blue', label='RadNet')
            plt.plot(torch.tensor(relu_loss).detach(), color='orange', label='ReLUNet')

        else:
            plt.plot(torch.tensor(rad_loss).detach(), color='blue')
            plt.plot(torch.tensor(relu_loss).detach(), color='orange')

        
    plt.title("Comparison of convergence rates")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('comparison.png')
    
    
main()

