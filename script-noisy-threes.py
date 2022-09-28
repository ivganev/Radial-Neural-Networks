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

    noisy_threes_sc2, _ = add_noise(
        label=3, n=int(num_samples), m=int(m_copies), verbose =False, noise_scale=1)

    noisy_threes_sc3, _ = add_noise(
        label=3, n=int(num_samples), m=int(m_copies), verbose =False, noise_scale=3)

    orig_threes = train_features[train_labels == 3]

    plt.subplot(2,3,1)
    plt.imshow(orig_threes[0][0], cmap="gray")

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left = False,
        labelbottom=False,
        labelleft=False)

    plt.subplot(2,3,2)
    plt.imshow(noisy_threes_sc2[0][0], cmap="gray")

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left = False,
        labelbottom=False,
        labelleft=False)

    plt.subplot(2,3,3)
    plt.imshow(noisy_threes_sc3[1][0], cmap="gray")

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left = False,
        labelbottom=False,
        labelleft=False)

    plt.subplot(2,3,4)
    plt.imshow(orig_threes[1][0], cmap="gray")

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left = False,
        labelbottom=False,
        labelleft=False)

    plt.subplot(2,3,5)
    plt.imshow(noisy_threes_sc2[m_copies][0], cmap="gray")

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left = False,
        labelbottom=False,
        labelleft=False)

    plt.subplot(2,3,6)
    plt.imshow(noisy_threes_sc3[m_copies+1][0], cmap="gray")

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left = False,
        labelbottom=False,
        labelleft=False)

    #plt.savefig('noisy_threes.png')
    plt.show()

    
main()