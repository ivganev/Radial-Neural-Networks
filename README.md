# Experiments for: Universal approxmiation and model compression for radial neural networks (NeurIPS 2022 Submission)

This repository accompanies the submission our NeurIPS May 2022 Submission. This code was adapted from this repo: https://github.com/ivganev/QR-decomposition-radial-NNs. The terminology and formulation of the current paper differs from that of the old paper and we have not updated everything. The experiments are still valid. 

## The file source.py

### The representation class

We create a class for the parameter space of an MLP with L layers.  

- The reduced representation R = W^{red}

- The transformed representation Q^{-1}W.

### Radial neural networks

We use PyTorch modules to implement radial activation functions in the class RadAct where there is an option to add a shift. We then implement radial neural networks in the class RadNet which has methods to:

- Set the weights.

- Export the weights 'W'.

- Export the reduced network (with weights R from the QR decomposition for the weights W).

- Export the transformed network (with weights Q^{-1} W where Q is from the QR decomposition for W).

### Training

For training models, we have three different types of training loops:

- A basic training loop with usual gradient descent. There is no optimizer in order to remove randomness. 

- A training loop with projected gradient descent. We define the appropriate masks in order to implement this properly. 

- A training loop with usual gradient descent and a stopping value for the loss function

## The file script-experiment-1-and-2.py

In this experiment, we instantiate a radial neural network with weights W and show that projected gradient descent on the transformed network (with weights Q^{-1} W) matches usual gradient descent on the reduced network (with weights R). Specifically, the values of the loss function are the same in both training regimes, epoch by epoch.  We also check the neural functions of f_W and f_R match.

    python script-experiment-6-1-and-6-2.py

## The file script-experiment-3.py

In this experiment, we instantiate a radial neural network with weights W and a somewhat large dimension vector. We train both the original model and the reduced model (with weights R coming from the QR decomposition of W) with usual gradient descent using a stopping value for the loss function. We show that the reduced model achieves this low value for the loss function after less time (albeit after more epochs).

    python script-experiment-6-3.py
