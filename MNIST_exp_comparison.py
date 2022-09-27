import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from source import *
from MNIST_source import *

from statistics import mean, stdev

torch.manual_seed(1)

def main():
    
    # Number of original images
    n = 3
    
    # Number of noisy samples for each original sample
    m = 10
    
    # Widths of both neural networks
    widths = [28*28, 28*28 + 1, 28*28 + 2, n]
    
    # Number of epochs
    num_epochs = 100

    # Number of trials
    num_trials = 2
    
    radnet_final_losses = []
    radnet_final_accuracies = []

    relunet_final_losses = []
    relunet_final_accuracies = []

    for trial in tqdm(range(num_trials)):
        rad_los, rad_acc, relu_los, relu_acc = train_both(
            num_samples = n,
            m_copies = m,
            dim_vector= widths,
            verbose=False,
            num_epochs=num_epochs)
        radnet_final_losses.append(round(rad_los[-1].item(),5))
        radnet_final_accuracies.append(rad_acc[-1].item())
        relunet_final_losses.append(round(relu_los[-1].item(),5))
        relunet_final_accuracies.append(relu_acc[-1].item())
        
    print("")
    print("Over %d trials, each training for %d epochs:" % (num_trials, num_epochs))
    print("")

    print("Radnet Loss = {1:.3e} +/- {2:.3e}".
        format(radnet_final_losses, mean(radnet_final_losses), stdev(radnet_final_losses))
    )

    print("Radnet Accuracy = {1:.3e} +/- {2:.3e}".
        format(radnet_final_accuracies, mean(radnet_final_accuracies), stdev(radnet_final_accuracies))
    )

    print("ReLU MLP Loss = {1:.3e} +/- {2:.3e}".
        format(relunet_final_losses, mean(relunet_final_losses), stdev(relunet_final_losses))
    )

    print(
        "ReLU MLP Accuracy = {1:.3e} +/- {2:.3e}".
        format(relunet_final_accuracies, mean(relunet_final_accuracies), stdev(relunet_final_accuracies))
    )
        

main()