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

torch.manual_seed(1)
np.random.seed(1)

def main():
    num_samples = 2
    m_copies = 2
    noise_scale = 3

    noisy_threes, noisy_labels = add_noise(
        label=3, n=int(num_samples), m=int(m_copies), verbose =False, noise_scale=noise_scale)

    orig_threes = train_features[train_labels == 3]

    plt.subplot(2,3,1)
    plt.imshow(orig_threes[0][0], cmap="gray")
    plt.subplot(2,3,2)
    plt.imshow(noisy_threes[0][0], cmap="gray")
    plt.subplot(2,3,3)
    plt.imshow(noisy_threes[1][0], cmap="gray")

    plt.subplot(2,3,4)
    plt.imshow(orig_threes[1][0], cmap="gray")
    plt.subplot(2,3,5)
    plt.imshow(noisy_threes[m_copies][0], cmap="gray")
    plt.subplot(2,3,6)
    plt.imshow(noisy_threes[m_copies+1][0], cmap="gray")
    #plt.savefig('noisy_threes.png')
    plt.show()
    
main()