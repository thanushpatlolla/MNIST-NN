import numpy as np
from activations import ReLU
from loss import CrossEntropyLoss as loss
from plot import Plot
from optimizer import SGD, AdamW


#mini-batch with const learning rate, He initialization
#code for math written by me, plotting code from ai, wrote dataset loader
    
class NN:
    #default hyperparamters are optimized for mnist
    def __init__(self, training_set, test_data, input_dim, epochs=25, hidden_dims=[256, 128], 
                 output_dim=10, act=ReLU(), lr_max=1e-3, lr_min=1e-5, warmup=2,
                 batch_size=128, optimizer=AdamW, weight_decay=7.5e-3,dropout=0.3):
        
        self.x_train, self.y_train=training_set
        self.test_data=test_data
        self.input_dim=input_dim
        self.epochs=epochs
        self.hidden_dims=hidden_dims
        self.output_dim=output_dim
        self.act=act
        self.batch_size=batch_size
        self.dims=[input_dim]+list(hidden_dims)+[output_dim]
        self.split_data()
        self.plot=Plot()
        self.warmup=warmup
        self.lr_max=lr_max
        self.lr_min=lr_min
        self.optimizer=optimizer(self.lr_max, weight_decay=weight_decay)
        self.t=0
        self.warmup_steps=int(self.warmup*len(self.x_train)//self.batch_size)
        self.dropout=dropout
        print("nn initialized")
    
    #randomly choose half to be validation set
    def split_data(self):
        n=len(self.test_data[0])
        indices = np.arange(n)
        np.random.shuffle(indices)
        split=n//2
        val_i = indices[:split]
        test_i = indices[split:]
        self.x_val, self.y_val=self.test_data[0][val_i], self.test_data[1][val_i]
        self.x_test, self.y_test=self.test_data[0][test_i], self.test_data[1][test_i]
    
    #create relevant arrays
    def initialize(self):
        self.weights=[]
        self.biases=[]
        self.activations=[np.zeros(self.dims[0], dtype=np.float32)]
        self.preactivations=[] #stores z in z=Wa+b, inputs to act functions
        for i in range(len(self.dims)-1):
            #He initialization
            self.weights.append(np.random.normal(0.0, np.sqrt(2/self.dims[i]), (self.dims[i+1], self.dims[i])).astype(np.float32))
            self.biases.append(np.zeros(self.dims[i+1]).astype(np.float32))
            self.activations.append(np.zeros(self.dims[i+1]).astype(np.float32))
            self.preactivations.append(np.zeros(self.dims[i+1]).astype(np.float32))
        
        #using glorot initialization for the final layer (bc its softmax not relu)
        self.weights[-1]=np.random.normal(0.0, np.sqrt(2/(self.dims[-1]+self.dims[-2])), (self.dims[-1], self.dims[-2])).astype(np.float32)
            
            
    def forward_pass(self, input):
        self.activations[0]=input
        self.mask=[]
        for i in range(0, len(self.dims)-2):
            self.preactivations[i]=self.activations[i]@self.weights[i].T+self.biases[i]
            self.activations[i+1]=self.act.act(self.preactivations[i])
            self.mask.append(np.random.binomial(1, 1-self.dropout, self.activations[i+1].shape))
            if self.dropout>0:
                self.activations[i+1]=self.activations[i+1]*self.mask[i]/(1-self.dropout)
        
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
            if self.dropout>0:
                derivs[-1]=derivs[-1]*self.mask[i-1]/(1-self.dropout)
            
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
        
            
    def accuracy(self, x_data, y_data):
        ans=self.evaluate(x_data)
        return np.mean(np.argmax(ans, axis=1) == y_data)
    
    
    def create_batches(self):
        batches=[]
        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)
        for i in range(0, len(self.x_train), self.batch_size):
            batch_i=indices[i:i+self.batch_size]
            batches.append((self.x_train[batch_i], self.y_train[batch_i]))
        return batches
    
    def lr_schedule(self):
        total_steps = self.epochs * np.ceil(len(self.x_train) / self.batch_size)
        if self.t <= self.warmup_steps:
            ans = self.lr_max*self.t/self.warmup_steps
        else:
            ans = self.lr_min+0.5*(self.lr_max-self.lr_min)*(1+np.cos(np.pi*(self.t-self.warmup_steps)/max(1, total_steps-self.warmup_steps)))
            
        self.t+=1
        return ans

    
    def train(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.initialize()
        for _ in range(self.epochs):
            self.batches=self.create_batches()
            for x_batch, y_batch in self.batches:
                self.forward_pass(x_batch)
                wgrads, bgrads=self.backwards_pass(y_batch)  
                self.optimizer.update_lr(self.lr_schedule())
                self.optimizer.step(self.weights, self.biases, wgrads, bgrads)
                
            self.train_losses.append(self.loss(self.x_train, self.y_train))
            self.val_losses.append(self.loss(self.x_val, self.y_val))
            self.train_accuracies.append(self.accuracy(self.x_train, self.y_train))
            self.val_accuracies.append(self.accuracy(self.x_val, self.y_val))
            self.plot.plot(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)
            
                
        self.plot.final_plot(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

        print("Final Training loss: ", self.train_losses[-1])
        print("Final Validation loss: ", self.val_losses[-1])
        print("Final Training accuracy: ", self.train_accuracies[-1])
        print("Final Validation accuracy: ", self.val_accuracies[-1])
        print("Final Test accuracy: ", self.accuracy(self.x_test, self.y_test))
        print("Training complete")
        return