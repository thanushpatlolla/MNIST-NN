import numpy as np

class SGD:
    def __init__(self, learning_rate, l2_reg=0.0):
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
    def step(self, weights, biases, wgrads, bgrads):
        for i in range(len(weights)):
            #l2 reg if we want it (we add 1/2lambda*w^2 to the loss so lambda*w to grad)
            weights[i]-=self.learning_rate*(wgrads[i] + self.l2_reg * weights[i])
            biases[i]-=self.learning_rate*bgrads[i]
    
    def update_lr(self, new_lr):
        self.learning_rate = new_lr


class AdamW:
    def __init__(self, learning_rate, weight_decay, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.wd = weight_decay
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.b1_pow = 1.0
        self.b2_pow = 1.0
        self.m_w = None; self.v_w = None
        self.m_b = None; self.v_b = None
    
    def update_lr(self, new_lr):
        self.lr = new_lr

    def step(self, weights, biases, wgrads, bgrads):
        self.t += 1
        self.b1_pow *= self.b1
        self.b2_pow *= self.b2
        one_minus_b1_pow = 1.0 - self.b1_pow
        one_minus_b2_pow = 1.0 - self.b2_pow

        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            # weights
            mw = self.m_w[i] = self.b1 * self.m_w[i] + (1 - self.b1) * wgrads[i]
            vw = self.v_w[i] = self.b2 * self.v_w[i] + (1 - self.b2) * (wgrads[i] * wgrads[i])
            mw_hat = mw / one_minus_b1_pow
            vw_hat = vw / one_minus_b2_pow
            denom = np.sqrt(vw_hat) + self.eps

            # Adam step
            weights[i] -= self.lr * (mw_hat / denom + self.wd * weights[i])

            # biases (no decay)
            mb = self.m_b[i] = self.b1 * self.m_b[i] + (1 - self.b1) * bgrads[i]
            vb = self.v_b[i] = self.b2 * self.v_b[i] + (1 - self.b2) * (bgrads[i] * bgrads[i])
            mb_hat = mb / one_minus_b1_pow
            vb_hat = vb / one_minus_b2_pow
            biases[i] -= self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))