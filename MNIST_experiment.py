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


def main():
    
    print("Test run")
    train_both(
        num_samples = 3,
        m_copies = 10,
        dim_vector= [28*28, 2, 2, 2, 1])
    
    print(" ")
    
    
    ns = [3,4,5]
    ms = [100,500,1000]
    d= 28*28
    dim_vecs = [
        [d, d+1, d+2, d+3, 1],
        [d, d+1, d+2, d+3, d+4, 1],
        [d, d+1, d+2, d+3, d+4, d+5, 1]]
    
    # Only have train_both coded for four-layer ReLU networks
    
    if False:
        for n in ns:
            for m in ms:
                train_both(
                    num_samples = n,
                    m_copies = m,
                    dim_vector= [d, d+1, d+2, d+3, d+4, 1])
            
    
    return


main()