import numpy as np

class ReLU():
    def act(self, x):
        return np.maximum(0,x)

    def dact(self, x):
        return (x>0).astype(x.dtype)