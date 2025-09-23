import numpy as np
import matplotlib.pyplot as plt
import struct
from os.path  import join

#we use label smoothing with 0.9 for the one-hot encoding, mini-batch with const learning rate, He initialization

class ReLU():
    def act(self, x):
        return np.maximum(0,x)

    def dact(self, x):
        return (x>0).astype(x.dtype)
    
class GeLU():
    def act(self, x):
        return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))

    def dact(self, x):
        return #do this later

def softmax(xs):
    exps = np.exp(xs - np.max(xs))
    return exps / np.sum(exps)
    
    
class NN:
    def __init__(self, training_set, test_data, input_dim, hidden_dims, output_dim, act=ReLU(), learning_rate=0.01, batch_size=32):
        self.x_train, self.y_train=training_set
        self.test_data=test_data
        self.input_dim=input_dim
        self.hidden_dims=hidden_dims
        self.output_dim=output_dim
        self.act=act
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.dims=[input_dim]+list(hidden_dims)+[output_dim]
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
        self.fig = None
        self.ax = None
        self.step = 0
        self.plot_every = 16


        
                
    
    def initialize(self):
        self.weights=[]
        self.biases=[]
        self.activations=[np.zeros(self.dims[0])]
        self.preactivations=[] #stores z in z=Wa+b, inputs to act functions
        for i in range(len(self.dims)-1):
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
    
    def backwards_pass(self, output):
        weightgrads=[]
        biasgrads=[]
        y=np.full(self.output_dim, 0.1)
        y[output]=0.9
        derivs=[self.activations[-1]-y] #derivative of loss wrt to input to the layer, so partial L partial z_i where a_i=relu(z_i)
        for i in reversed(range(1, len(self.dims)-1)):
            biasgrads.append(derivs[-1])
            weightgrads.append(np.outer(derivs[-1], self.activations[i]))
            derivs.append((self.weights[i].T@derivs[-1])*self.act.dact(self.preactivations[i-1]))
            
        biasgrads.append(derivs[-1])
        weightgrads.append(np.outer(derivs[-1], self.activations[0]))
        weightgrads.reverse()
        biasgrads.reverse()
        return (weightgrads, biasgrads)
            
    def evaluate(self, input):
        ans=input
        for i in range(len(self.dims)-2):
            ans=self.act.act(self.weights[i]@ans+self.biases[i])
            
        ans=softmax(self.weights[len(self.dims)-2]@ans+self.biases[len(self.dims)-2])
        
        return ans
            
                                        
            
    def loss(self, x_data, y_data):
        loss=0.0
        for i in range(len(y_data)):
            ans=self.evaluate(x_data[i])
            for j in range(self.output_dim):
                if y_data[i]==j:
                    loss-=0.9*np.log(ans[j]+1e-12)
                else:
                    loss-=0.1*np.log(ans[j]+1e-12)
        
        return loss
                
    def plot(self):
        #this code is ai generated, I don't care to write plotting code
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
        self.train_losses.append(self.loss(self.x_train, self.y_train))
        self.val_losses.append(self.loss(self.x_val, self.y_val))
        self.ax.clear()
        self.ax.plot(self.train_losses, label="train loss")
        self.ax.plot(self.val_losses, label="val loss")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training & Validation Loss")
        self.ax.legend()
        self.fig.canvas.draw_idle()
        plt.pause(0.01)  # let the plot refresh


    def optimize_step(self, batch):
        wgrads = [np.zeros_like(W) for W in self.weights]
        bgrads = [np.zeros_like(b) for b in self.biases]
        for x,y in batch:
            self.forward_pass(x)
            wgrad, bgrad=self.backwards_pass(y)        
            for i in range(len(self.weights)):
                wgrads[i] += wgrad[i]
                bgrads[i] += bgrad[i]
        
        for i in range(len(self.dims)-1):
            self.weights[i]-=self.learning_rate*wgrads[i]/len(batch)
            self.biases[i]-=self.learning_rate*bgrads[i]/len(batch)
        self.step += 1
        if self.step % self.plot_every == 0:
            self.plot()
            
        self.plot()
        
            

        

    def train(self, epochs):
        self.train_losses = []
        self.val_losses = []
        plt.ion()  # turn on interactive mode
        self.initialize()
        for _ in range(epochs):
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            for i in range(0, len(self.x_train), self.batch_size):
                batch_i=indices[i:i+self.batch_size]
                self.optimize_step(list(zip(self.x_train[batch_i], self.y_train[batch_i])))
                
            
                
                
        plt.ioff()   # turn interactive mode back off
        plt.show()   # keep the final figure open


#
# Below code is from the kaggle website
#

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.frombuffer(file.read(), dtype=np.uint8)
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.frombuffer(file.read(), dtype=np.uint8)
        images = image_data.reshape(size, rows * cols).astype(np.float32) / 255.0
        return images, labels
             
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
input_path = 'input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
training_set, test_data = mnist_dataloader.load_data()

MNIST_solver=NN(training_set, test_data, len(training_set[0][0]), [128, 64], 10, learning_rate=0.005)
MNIST_solver.train(50)