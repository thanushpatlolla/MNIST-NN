import numpy as np
import matplotlib.pyplot as plt

#we use label smoothing with 0.9 for the one-hot encoding, mini-batch with const learning rate, He initialization

class ReLU():
    def act(x):
        return np.maximum(0,x)

    def dact(x):
        return (x>0).astype(x.dtype)
    
class GeLU():
    def act(x):
        return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))

    def dact(x):
        return #do this later

def softmax(xs):
    exps = np.exp(xs - np.max(xs))
    return exps / np.sum(exps)

def dloss(logits, label):
    #gives the derivative of the cross-entropy loss wrt the logits (inputs) of the softmax function in the last layer
    #https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
    return logits-label
    
    
class NN:
    def __init__(self, data, input_dim, hidden_dims, output_dim, act=ReLU(), learning_rate=0.01, batch_size=32):
        self.data=data
        self.input_dim=input_dim
        self.hidden_dims=hidden_dims
        self.output_dim=output_dim
        self.act=act
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        
        self.dims=[input_dim]+list(hidden_dims)+[output_dim]
        
    def initialize(self):
        self.weights=[]
        self.biases=[]
        self.activations=[np.zeros(self.dims[0])]
        self.preactivations=[] #stores z in z=Wa+b, inputs to act functions
        for i in range(len(self.dims-1)):
            self.weights.append(np.random.normal(0.0, np.sqrt(2/self.dims[i]), (self.dims[i+1], self.dims[i])))
            self.biases.append(np.zeros(self.dims[i+1]))
            self.activations.append(np.zeros(self.dims[i+1]))
            self.preactivations.append(np.zeros(self.dims[i+1]))
            
            
    def forward_pass(self, input):
        self.activations[0]=input
        for i in range(0, len(self.dims)-2):
            self.preactivations[i]=self.weights[i]@self.activations[i]+self.biases[i]
            self.activations[i+1]=self.act.act(self.preactivations[i])
        
        #because mnist is classification into 10 diff things, 
        #could probably make the final output layer configurable at some point
        self.preactivations[len(self.dims)-2]=self.weights[len(self.dims)-2]@self.activations[len(self.dims)-2]+self.biases[len(self.dims)-2]
        self.activations[len(self.dims)-1]=softmax(self.preactivations[len(self.dims)-2])
    
    def backwards_pass(self):
        weightgrads=[]
        biasgrads=[]
        derivs=[dloss(self.preactivations[len(self.dims)-2])] #derivative of loss wrt to input to the layer, so partial L partial z_i where a_i=relu(z_i)
        for i in reversed(range(1, len(self.dims)-1)):
            biasgrads.append(derivs[len(self.dims)-2-i])
            weightgrads.append(derivs[i]@self.activations[i].T)
            derivs.append((self.weights[i].T@derivs[len(self.dims)-2-i])*self.act.dact(self.activations[i]))
            
        biasgrads[len(self.dims)-2]=self.derivs[len(self.dims)-2]
        weightgrads[len(self.dims)-2]=self.derivs[len(self.dims)-2]@self.activations[0].T
        weightgrads.reverse()
        biasgrads.reverse()
        return (weightgrads, biasgrads)
            
    def optimize_step(self, batch):
        wgrads=[]
        bgrads=[]
        for input in batch:
            self.forward_pass(input)
            wgrad, bgrad=self.backwards_pass()        
            wgrads.append(wgrad)
            bgrads.append(bgrad)
        
        avg_wgrads = np.mean(np.stack(wgrads, axis=0), axis=0)
        avg_bgrads  = np.mean(np.stack(bgrads, axis=0), axis=0)
        
        for i in range(len(self.dims)-1):
            self.weights[i]-=self.learning_rate*avg_wgrads[i]
            self.bgrads[i]-=self.learning_rate*avg_bgrads[i]
            
            
    def train()
        
        

        

            
            
              