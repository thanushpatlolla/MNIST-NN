import numpy as np
from output_layer import softmax as out

class Evals:
    def __init__(self, act):
        self.act=act
    def evaluate(self, weights, biases, input):
        #get the output
        ans=input
        for i in range(len(weights)-1):
            ans=self.act.act(ans@weights[i].T+biases[i])
            
        ans=out.output(ans@weights[len(weights)-1].T+biases[len(weights)-1])
        
        return ans
                                             
            
    def loss(self, weights, biases, x_data, y_data):
        ans=self.evaluate(weights, biases, x_data)
        y=np.eye(ans.shape[1])[y_data]
        
        return -np.mean(np.sum(y*np.log(ans+1e-12), axis=1))
        
            
    def accuracy(self, weights, biases, x_data, y_data):
        ans=self.evaluate(weights, biases, x_data)
        return np.mean(np.argmax(ans, axis=1) == y_data)
