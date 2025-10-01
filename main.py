from dataloader import DataLoader
from nn import NN
from activations import ReLU, GeLU
from optimizer import AdamW
from defaults import defaults

#uses the hyperparameters that seem to work best for each dataset, stored in defaults.py
#the nn can be stored in a variable and then used for more evaluation


MNIST_solver=defaults.mnist()
CIFAR10_solver=defaults.cifar10()