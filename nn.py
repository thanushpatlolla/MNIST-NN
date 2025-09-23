import numpy as np
import matplotlib.pyplot as plt
import struct
from os.path  import join
from tqdm import tqdm
import time

#mini-batch with const learning rate, He initialization
#code for math written by me, plotting code from ai, dataset reading from kaggle

class ReLU():
    def act(self, x):
        return np.maximum(0,x)

    def dact(self, x):
        return (x>0).astype(x.dtype)
    
def softmax(xs):
    exps = np.exp(xs - np.max(xs, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
    
    
class NN:
    def __init__(self, training_set, test_data, input_dim, hidden_dims=[128, 64], output_dim=10, act=ReLU(), learning_rate=0.01, batch_size=32):
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
        self.activations[len(self.dims)-1]=softmax(self.preactivations[len(self.dims)-2])
    
    def backwards_pass(self, y_batch):
        B=y_batch.shape[0]
        weightgrads=[]
        biasgrads=[]
        y=np.eye(self.output_dim)[y_batch]
        derivs=[self.activations[-1]-y] #derivative of loss wrt to input to the layer, so partial L partial z_i where a_i=relu(z_i)
        for i in reversed(range(1, len(self.dims)-1)):
            biasgrads.append(derivs[-1].sum(axis=0)/B)
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
            
        ans=softmax(ans@self.weights[len(self.dims)-2].T+self.biases[len(self.dims)-2])
        
        return ans
            
                                        
            
    def loss(self, x_data, y_data):
        ans=self.evaluate(x_data)
        y=np.eye(self.output_dim)[y_data]
        
        return -np.mean(np.sum(y*np.log(ans+1e-12), axis=1))
                
    def plot(self):
        if self.ax is None or not hasattr(self, 'ax2'):
            self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))

        self.ax.clear()
        self.ax.plot(self.train_losses, label="train loss")
        self.ax.plot(self.val_losses, label="val loss")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Loss")
        self.ax.legend()

        self.ax2.clear()
        self.ax2.plot(self.train_accuracies, label="train acc")
        self.ax2.plot(self.val_accuracies, label="val acc")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Accuracy")
        self.ax2.set_title("Accuracy")
        self.ax2.set_ylim(0.8, 1.0)
        self.ax2.legend()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)


    def optimize_step(self, x_batch, y_batch):
        self.forward_pass(x_batch)
        wgrads, bgrads=self.backwards_pass(y_batch)        
        for i in range(len(self.dims)-1):
            self.weights[i]-=self.learning_rate*wgrads[i]
            self.biases[i]-=self.learning_rate*bgrads[i]
            
        self.step+=1
        if self.step % 100 == 0:
            self.train_losses.append(self.loss(x_batch, y_batch))
            self.val_losses.append(self.loss(self.x_val, self.y_val))
            self.plot()
            
            
    def accuracy(self, x_data, y_data):
        ans=self.evaluate(x_data)
        y=np.eye(self.output_dim)[y_data]
        return np.mean(np.argmax(ans, axis=1) == np.argmax(y, axis=1))
    
    def train(self, epochs):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        plt.ion()  # turn on interactive mode
        self.initialize()
        for _ in range(epochs):
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            for i in range(0, len(self.x_train), self.batch_size):
                batch_i=indices[i:i+self.batch_size]
                self.optimize_step(self.x_train[batch_i], self.y_train[batch_i])
                
            self.train_accuracies.append(self.accuracy(self.x_train, self.y_train))
            self.val_accuracies.append(self.accuracy(self.x_val, self.y_val))
            self.plot()
                
            
                
                
        plt.ioff()
        self.plot()
        plt.show()

        print("Final Training loss: ", self.train_losses[-1])
        print("Final Validation loss: ", self.val_losses[-1])
        print("Final Training accuracy: ", self.accuracy(self.x_train, self.y_train))
        print("Final Validation accuracy: ", self.accuracy(self.x_val, self.y_val))
        print("Final Test accuracy: ", self.accuracy(self.x_test, self.y_test))
        print("Training complete")
        return


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

MNIST_solver=NN(training_set, test_data, len(training_set[0][0]))
MNIST_solver.train(50)