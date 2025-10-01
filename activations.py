import numpy as np

class ReLU():
    @staticmethod
    def act(x):
        return np.maximum(0,x)


    @staticmethod
    def dact(x):
        return (x>0).astype(x.dtype)
    
class GeLU:
    @staticmethod
    def act(x):
        return 0.5*x*(1.0+np.tanh(
            np.sqrt(2/np.pi)*(x+0.044715*x**3)
        ))
        
    @staticmethod
    def dact(x):
        tanh_term = np.tanh(
            np.sqrt(2/np.pi)*(x+0.044715*x**3)
        )
        left = 0.5*(1.0+tanh_term)
        right = 0.5*x*(1-tanh_term**2)*np.sqrt(2/np.pi)*(1+3*0.044715*x**2)
        return left+right
