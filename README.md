# Experiments for: Universal approximation and model compression for radial neural networks

The python files in this folder accompany the manuscript 'Universal approximation and model compression for radial neural networks'.    

## The file source.py

### The representation class

We create a class ```representation``` for the parameter space of an MLP with L layers. (The terminology comes from another project.) This class implements Algorithm 1 as the method ```QR_decomposition```. Furthermore, one can access:

- The reduced representation R = W^{red}

- The orthogonal matrices Q

- The transformed representation Q^{-1}W.

### Radial neural networks

We use PyTorch modules to implement radial activation functions in the class RadAct where there is an option to add a shift. We then implement radial neural networks in the class RadNet which has methods to:

- Set the weights.

- Export the weights 'W'.

- Export the reduced network (with the reduced weights from the QR-compression algorithm for the weights W).

- Export the transformed network (with weights Q^{-1} W where Q is from the QR decomposition for W).

### Training

For training models, we have three different types of training loops:

- A basic training loop with usual gradient descent. There is no optimizer in order to remove randomness. 

- A training loop with projected gradient descent. We define the appropriate masks in order to implement this properly. 

- A training loop with usual gradient descent and a stopping value for the loss function

## The file script-experiment-1-and-2.py

In this experiment, we instantiate a radial neural network with weights W and show that projected gradient descent on the transformed network (with weights Q^{-1} W) matches usual gradient descent on the compressed network. Specifically, the values of the loss function are the same in both training regimes, epoch by epoch.  We also check the neural functions of the original and compressed networks match.

    python script-experiment-1-and-2.py

## The file script-experiment-3.py

In this experiment, we instantiate a radial neural network with weights W and a somewhat large dimension vector. We train both the original model and the reduced model (with compressed weights coming from the QR-compress algorithm applied to W) with usual gradient descent using a stopping value for the loss function. We show that the reduced model achieves this low value for the loss function after less time (albeit after more epochs).

    python script-experiment-3.py


## The file script-experiment-4.py

Noisy image recovery, comparision between a step-ReLU radial network and a ReLU MLP.

    python script-experiment-4.py


## The file script-experiment-5.py

Convergence rate comparision in noisy image recovery.

    python script-experiment-5.py