import numpy as np
import struct
from activations import ReLU
from loss import CrossEntropyLoss as loss
from plot import Plot


#mini-batch with const learning rate, He initialization
#code for math written by me, plotting code from ai, wrote dataset loader
    
class NN:
    def __init__(self, training_set, test_data, input_dim, hidden_dims=[128, 64], output_dim=10, act=ReLU(), learning_rate=0.001, batch_size=32, l2_reg=0.0):
        self.x_train, self.y_train=training_set
        self.test_data=test_data
        self.input_dim=input_dim
        self.hidden_dims=hidden_dims
        self.output_dim=output_dim
        self.act=act
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.l2_reg=l2_reg
        self.dims=[input_dim]+list(hidden_dims)+[output_dim]
        #test/val split should be a separate function
        x_test_data, y_test_data=self.test_data
        n=len(x_test_data)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split=n//2
        val_i = indices[:split]
        test_i = indices[split:]
        self.x_val, self.y_val=x_test_data[val_i], y_test_data[val_i]
        self.x_test, self.y_test=x_test_data[test_i], y_test_data[test_i]
        # plotting state
        self.plot = Plot()
    
    def initialize(self):
        self.weights=[]
        self.biases=[]
        self.activations=[np.zeros(self.dims[0], dtype=np.float32)]
        self.preactivations=[] #stores z in z=Wa+b, inputs to act functions
        for i in range(len(self.dims)-1):
            self.weights.append(np.random.normal(0.0, np.sqrt(2/self.dims[i]), (self.dims[i+1], self.dims[i])).astype(np.float32))
            self.biases.append(np.zeros(self.dims[i+1]).astype(np.float32))
            self.activations.append(np.zeros(self.dims[i+1]).astype(np.float32))
            self.preactivations.append(np.zeros(self.dims[i+1]).astype(np.float32))
        
        #using glorot initialization for the final layer
        self.weights[-1]=np.random.normal(0.0, np.sqrt(2/(self.dims[-1]+self.dims[-2])), (self.dims[-1], self.dims[-2]))
            
            
    def forward_pass(self, input):
        self.activations[0]=input
        for i in range(0, len(self.dims)-2):
            self.preactivations[i]=self.activations[i]@self.weights[i].T+self.biases[i]
            self.activations[i+1]=self.act.act(self.preactivations[i])
        
        #because mnist is classification into 10 diff things, 
        #could probably make the final output layer configurable at some point
        self.preactivations[len(self.dims)-2]=self.activations[len(self.dims)-2]@self.weights[len(self.dims)-2].T+self.biases[len(self.dims)-2]
        self.activations[len(self.dims)-1]=loss.softmax(self.preactivations[len(self.dims)-2])
    
    def backwards_pass(self, y_batch):
        B=y_batch.shape[0]
        weightgrads=[]
        biasgrads=[]
        y=np.eye(self.output_dim)[y_batch]
        derivs=[self.activations[-1]-y] #derivative of loss wrt to input to the layer, so partial L partial z_i where a_i=relu(z_i)
        for i in reversed(range(1, len(self.dims)-1)):
            biasgrads.append(np.mean(derivs[-1], axis=0))
            weightgrads.append(derivs[-1].T@self.activations[i]/B)
            derivs.append((derivs[-1]@self.weights[i])*self.act.dact(self.preactivations[i-1]))
            
        biasgrads.append(derivs[-1].sum(axis=0)/B)
        weightgrads.append(derivs[-1].T@self.activations[0]/B)
        weightgrads.reverse()
        biasgrads.reverse()
        return (weightgrads, biasgrads)
            
    def evaluate(self, input):
        ans=input
        for i in range(len(self.dims)-2):
            ans=self.act.act(ans@self.weights[i].T+self.biases[i])
            
        ans=loss.softmax(ans@self.weights[len(self.dims)-2].T+self.biases[len(self.dims)-2])
        
        return ans
            
                                        
            
    def loss(self, x_data, y_data):
        ans=self.evaluate(x_data)
        y=np.eye(self.output_dim)[y_data]
        
        return -np.mean(np.sum(y*np.log(ans+1e-12), axis=1))
                

    def optimize_step(self, x_batch, y_batch):
        self.forward_pass(x_batch)
        wgrads, bgrads=self.backwards_pass(y_batch)        
        for i in range(len(self.dims)-1):
            #l2 reg if we want it
            wgrads[i] += 2.0*self.l2_reg * self.weights[i]
            self.weights[i]-=self.learning_rate*wgrads[i]
            self.biases[i]-=self.learning_rate*bgrads[i]
            
            
            
    def accuracy(self, x_data, y_data):
        ans=self.evaluate(x_data)
        return np.mean(np.argmax(ans, axis=1) == y_data)
    
    def train(self, epochs):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.initialize()
        for _ in range(epochs):
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            for i in range(0, len(self.x_train), self.batch_size):
                batch_i=indices[i:i+self.batch_size]
                self.optimize_step(self.x_train[batch_i], self.y_train[batch_i])
                
            self.train_losses.append(self.loss(self.x_train, self.y_train))
            self.val_losses.append(self.loss(self.x_val, self.y_val))
            self.train_accuracies.append(self.accuracy(self.x_train, self.y_train))
            self.val_accuracies.append(self.accuracy(self.x_val, self.y_val))
            self.plot.plot(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)
            
                
        self.plot.final_plot(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

        print("Final Training loss: ", self.train_losses[-1])
        print("Final Validation loss: ", self.val_losses[-1])
        print("Final Training accuracy: ", self.accuracy(self.x_train, self.y_train))
        print("Final Validation accuracy: ", self.accuracy(self.x_val, self.y_val))
        print("Final Test accuracy: ", self.accuracy(self.x_test, self.y_test))
        print("Training complete")
        return