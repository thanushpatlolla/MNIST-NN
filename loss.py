import numpy as np

class CrossEntropyLoss():
    
    @staticmethod
    def softmax(xs):
        exps = np.exp(xs - np.max(xs, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
