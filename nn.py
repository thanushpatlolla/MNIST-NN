import numpy as np

def ReLU(x):
    return np.maximum(0,x)

def GeLU(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
    
    
class NN:
    def __init__(self, input_dim, hidden_dims, output_dim, activation=ReLU, optimizer="mini-batch"):
        self.input_dim=input_dim
        self.hidden_dims=hidden_dims
        self.output_dim=output_dim
        #using He initialization so don't use stuff outside ReLU family
        self.activation=activation
        self.optimizer=optimizer
        