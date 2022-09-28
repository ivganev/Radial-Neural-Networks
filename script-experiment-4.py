import numpy as np
from typing import List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from source import *
from source_MNIST import *

from statistics import mean, stdev

torch.manual_seed(1)
np.random.seed(1)
device = 'cpu' # CUDA results are less reproducible

def main():
    
    #### Set hyperparameters
    
    # Number of original images
    n = 3
    
    # Number of noisy samples for each original sample
    m = 100
    
    # Widths of both neural networks
    widths = [28*28, 28*28 + 1, 28*28 + 2, n]
    
    # Number of epochs
    num_epochs = 150

    # Number of trials
    num_trials = 10
    
    # Noise level (>0.5 means there will be overlap)
    noise_scale = 3
    
    # Verbosity
    verbose = True
    
    #### Run trials
    
    metrics = ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
    metric_values = defaultdict(lambda: defaultdict(list))

    for trial in tqdm(range(num_trials)):
        logs = train_both(
            num_samples = n,
            m_copies = m,
            dim_vector = widths,
            verbose = verbose,
            epochs = num_epochs,
            noise_scale = noise_scale,
            device = device)
        for method, log in logs.items():
            for metric in metrics:
                metric_values[method][metric].append(log[metric][-1])
        
    print("")
    print("Over %d trials, each training for %d epochs:" % (num_trials, num_epochs))
    print("")

    for method, values in metric_values.items():
        for metric in metrics:
            print("{0} {1} = {2:.3g} +/- {3:.3e}".
                format(method, metric, mean(values[metric]), stdev(values[metric]))
            )

if __name__ == "__main__":
    main()

