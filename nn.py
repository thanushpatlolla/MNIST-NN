import numpy as np
import matplotlib.pyplot as plt
import struct
from os.path  import join

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
    
    def backwards_pass(self, output):
        weightgrads=[]
        biasgrads=[]
        derivs=[dloss(self.preactivations[len(self.dims)-2], output)] #derivative of loss wrt to input to the layer, so partial L partial z_i where a_i=relu(z_i)
        for i in reversed(range(1, len(self.dims)-1)):
            biasgrads.append(derivs[len(self.dims)-2-i])
            weightgrads.append(derivs[i]@self.activations[i].T)
            derivs.append((self.weights[i].T@derivs[len(self.dims)-2-i])*self.act.dact(self.activations[i]))
            
        biasgrads[len(self.dims)-2]=self.derivs[len(self.dims)-2]
        weightgrads[len(self.dims)-2]=self.derivs[len(self.dims)-2]@self.activations[0].T
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
            for j in range(10):
                loss-=y_data[i][j]*np.log(ans[j])
        
        return loss
                
    def plot(self):
        #this code is ai generated, I don't care to write plotting code
        self.train_losses = []
        self.val_losses = []

        fig, ax = plt.subplots()
        
        self.train_losses.append(self.loss(self.x_train, self.y_train))
        self.val_losses.append(self.loss(self.x_val, self.y_val))

        # --- live update plot ---
        ax.clear()
        ax.plot(self.train_losses, label="train loss")
        ax.plot(self.val_losses, label="val loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        plt.pause(0.01)  # let the plot refresh


    def optimize_step(self, batch):
        wgrads=[]
        bgrads=[]
        for pair in batch:
            self.forward_pass(pair[0])
            wgrad, bgrad=self.backwards_pass(pair[1])        
            wgrads.append(wgrad)
            bgrads.append(bgrad)
        
        avg_wgrads = np.mean(np.stack(wgrads, axis=0), axis=0)
        avg_bgrads  = np.mean(np.stack(bgrads, axis=0), axis=0)
        
        for i in range(len(self.dims)-1):
            self.weights[i]-=self.learning_rate*avg_wgrads[i]
            self.bgrads[i]-=self.learning_rate*avg_bgrads[i]
            
        self.plot()
        
            

        

    def train(self, epochs):
        plt.ion()  # turn on interactive mode
        for _ in range(epochs):
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            for i in range(0, len(self.x_train), self.batch_size):
                batch_i=indices[i:i+self.batch_size]
                self.optimize_step(zip(x_train[batch_i], y_train[batch_i]))
                
            
                
                
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
            labels = np.array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
input_path = '../input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
training_set, test_data = mnist_dataloader.load_data()

MNIST_solver=NN(training_set, test_data, len(training_set[0][0]), [128, 64], 10)
MNIST_solver.train(50)