#====================================================================
# Algorithm Definitions
#
# authors: Wei-Che Huang, YoungSeok Na
#====================================================================

import numpy as np
from tqdm.auto import tqdm
from scipy.special import expit
from sklearn.metrics import log_loss, mean_squared_error

#--------------------------------------------------------------------
# Global variable / Common functions
#--------------------------------------------------------------------
tqdmSwitch = False
weightEvalRes = 20000

# Sigmoid
def sigmoid(x):
    return expit(x)

# Bound values
def bound(x):
    x = np.where(x<1e-16,1e-16,x)
    return np.where(x>1-(1e-16), (1-1e-16), x)

# Log-loss prediction
def log_predict(w, x):
    return bound(sigmoid(x @ w))

# Mean-squared Error prediction
def mse_predict(w, x):
    return x @ w

# L1 (Lasso) regularization
def l1_regularize(w):
    return np.linalg.norm(w, ord=1)

def l1_grad(w):
    pass

# L2 (Ridge) regularization
def l2_regularize(w):
    return np.linalg.norm(w)

def l2_grad(lmda, w):
    return 2 * lmda * w

# placeholder regularization
def null_regularize(w):
    return 0

#--------------------------------------------------------------------
# SGD
#--------------------------------------------------------------------
class SGD:
    def __init__(self, reg=0, loss="log"):
        self.name = "SGD"
        
        # Default to log-loss
        if loss == "mse":
            self.pred = mse_predict
        else:
            self.pred = log_predict

        # Regularizer - default to L2
        if reg == 1:
            self.regularize = l1_regularize
            self.rgrad = l1_grad
        elif reg == 2:
            self.regularize = l2_regularize
            self.rgrad = l2_grad
        else:
            self.regularize = null_regularize
            self.rgrad = l2_grad

        self.cost = loss

    def run(self, x, y, step, iters, lmda=0, q=None):
        n = x.shape[0]
        d = x.shape[1]
        assert n == y.shape[0]
        y = y.reshape((n, 1))

        # Initialize weight
        w = np.zeros([d, 1])
        
        # Loss value
        costs = []

        # Fixed learning rate
        a = step

        # For each iteration
        for k in tqdm(range(iters), disable=tqdmSwitch):
            # Draw random sample with replacement
            idx = np.random.randint(0, n)
            xx = x[idx]
            yy = y[idx]
            
            # Make prediction
            y_pred = self.pred(w, xx)

            # Update weights
            grad = (xx * (y_pred - yy)).reshape((d, 1))
            w = w - a * (grad + self.rgrad(lmda, w))

            # Compute cost
            if ((k + 1) % weightEvalRes == 0):
                pred = self.pred(w, x)
                if self.cost == "log":
                    costs += [log_loss(y, pred, labels=[0, 1]) + self.regularize(w)]
                elif self.cost == "mse":
                    costs += [mean_squared_error(y, pred, squared=False) / n + self.regularize(w)]
            
        if q != None:
            q.put([w, np.array(costs)])
            
        return w, np.array(costs)

#--------------------------------------------------------------------
# SAG
#--------------------------------------------------------------------
class SAG:
    def __init__(self, reg=0, loss="log"):
        self.name = "SAG"

        # Default to log-loss
        if loss == "mse":
            self.pred = mse_predict
        else:
            self.pred = log_predict

        # Regularizer - default to L2
        if reg == 1:
            self.regularize = l1_regularize
            self.rgrad = l1_grad
        elif reg == 2:
            self.regularize = l2_regularize
            self.rgrad = l2_grad
        else:
            self.regularize = null_regularize
            self.rgrad = l2_grad

        self.cost = loss

    def run(self, x, y, step, iters, lmda=0, q=None):
        n = x.shape[0]
        d = x.shape[1]
        assert n == y.shape[0]
        y = y.reshape((n, 1))

        # Initialize weights and gradient table
        w = np.zeros([d, 1])
        g = np.zeros((n, d, 1)) # Gradient table
        G = np.zeros_like(w) # Gradient table sum

        # Sampling
        idxs = []
        m = 0
        
        # Loss value
        costs = []

        # Fixed learning rate
        a = step
        
        # For each iteration
        for k in tqdm(range(iters), disable=tqdmSwitch):
            # Draw random sample with replacement
            idx = np.random.randint(0, n)
            xx = x[idx]
            yy = y[idx]
            
            # Make prediction
            y_pred = self.pred(w, xx)

            # Check if data point has been seen
            if idx not in idxs:
                idxs += [idx]
                m += 1
            
            # Calculate current gradient
            grad = (xx * (y_pred - yy)).reshape((d, 1))

            # Update gradient sum
            G = G - g[idx] + grad

            # Update previous sample gradient
            g[idx] = grad

            # Update weights
            w = w - a * (G / m + self.rgrad(lmda, w))

            #Compute cost
            if ((k + 1) % weightEvalRes == 0):
                pred = self.pred(w, x)
                if self.cost == "log":
                    costs += [log_loss(y, pred, labels=[0, 1]) + self.regularize(w)]
                elif self.cost == "mse":
                    costs += [mean_squared_error(y, pred, squared=False) / n + self.regularize(w)]
            
        if q != None:
            q.put([w, np.array(costs)])
            
        return w, np.array(costs)

#--------------------------------------------------------------------
# Finito
#--------------------------------------------------------------------
class Finito:
    def __init__(self, reg=0, loss="log"):
        self.name = "Finito"

        # Default to log-loss
        if loss == "mse":
            self.pred = mse_predict
        else:
            self.pred = log_predict

        # Regularizer - default to L2
        if reg == 1:
            self.regularize = l1_regularize
            self.rgrad = l1_grad
        elif reg == 2:
            self.regularize = l2_regularize
            self.rgrad = l2_grad
        else:
            self.regularize = null_regularize
            self.rgrad = l2_grad

        self.cost = loss

    def run(self, x, y, step, iters, lmda=0, q=None):
        n = x.shape[0]
        d = x.shape[1]
        assert n == y.shape[0]

        # Initialize weights (table) and gradient table
        w = np.zeros([d, 1])
        g = np.zeros((n, d, 1)) # Gradient table
        p = np.zeros((n, d, 1)) # Weight table
        G = np.zeros_like(w) # Gradient table sum
        P = np.zeros_like(w) # Weight table sum

        idx_range = np.arange(n)
        grad = np.zeros_like(w)
        idxs = []
        m = 0
        b = 0
        
        costs = []
        y = y.reshape((n, 1))
        
        # TODO: you can change this to whatever alpha value it is
        # - pg. 7 of Finito paper.
        alpha = 1

        #For each iteration
        for k in tqdm(range(iters), disable=tqdmSwitch):        
            # Learning rate - mu is convexity constant
            # a = 1/(alpha * mu * (k+1))
            a = step

            # Draw random sample with replacement
            # idx = np.random.randint(0,len(x))
            if (k % n) == 0:
                np.random.shuffle(idx_range)
            idx = idx_range[k % n]
            xx = x[idx]
            yy = y[idx]

            # Check if data point has been seen
            if idx not in idxs:
                idxs += [idx]
                m += 1

            # Make prediction
            y_pred = self.pred(w, xx)
            
            # Calculate current gradient
            grad = (xx * (y_pred - yy)).reshape((d, 1))

            #Update gradient and weight table
            G = G - g[idx] + grad
            P = P - p[idx] + w

            # Take the average of the weights \phi
            phi_bar = P / n
            # phi_bar = P / m

            # Take the average of the gradients
            g_bar = G / n
            # g_bar = G / m

            # Update weight
            w = phi_bar - a * g_bar + self.rgrad(lmda, w)

            #Update previous sample gradient and weight
            g[idx] = grad
            p[idx] = w

            #Compute cost
            if ((k + 1) % weightEvalRes == 0):
                pred = self.pred(w, x)
                if self.cost == "log":
                    costs += [log_loss(y, pred, labels=[0, 1]) + self.regularize(w)]
                elif self.cost == "mse":
                    costs += [mean_squared_error(y, pred, squared=False) / n + self.regularize(w)]

        if q != None:
            q.put([w, np.array(costs)])
            
        return w, np.array(costs)

#--------------------------------------------------------------------
# SAGA
#--------------------------------------------------------------------
class SAGA:
    def __init__(self, reg=0, loss="log"):
        # Default to log-loss
        if loss == "mse":
            self.pred = mse_predict
        else:
            self.pred = log_predict

        # Regularizer - default to L2
        if reg == 1:
            self.regularize = l1_regularize
            self.rgrad = l1_grad
        elif reg == 2:
            self.regularize = l2_regularize
            self.rgrad = l2_grad
        else:
            self.regularize = null_regularize
            self.rgrad = l2_grad

        self.cost = loss

    def run(self, x, y, step, iters, lmda=0, q=None):
        n = x.shape[0]
        d = x.shape[1]
        assert n == y.shape[0]

        #Initialize weights and bias
        w = np.zeros([d, 1])
        g = np.zeros((n, d, 1)) # Gradient table
        G = np.zeros_like(w) # Gradient table sum
        idxs = []
        m = 0
        
        costs = []
        y = y.reshape((n, 1))
        
        #Fixed learning rate
        a = step
        
        #For each iteration
        for k in tqdm(range(iters), disable=tqdmSwitch):
            
            #Draw random sample with replacement
            idx = np.random.randint(0, n)
            xx = x[idx]
            yy = y[idx]
            
            # Make prediction
            y_pred = self.pred(w, xx)

            # Check if data point has been seen
            if idx not in idxs:
                idxs += [idx]
                m += 1
            
            # Calculate current gradient
            grad = (xx * (y_pred - yy)).reshape((d, 1))

            # Update weights
            w = w - a*((grad - g[idx] + G / m) + self.rgrad(lmda, w))

            # Update gradient table
            G = G - g[idx] + grad

            # Update previous sample gradient
            g[idx] = grad

            #Compute cost
            if ((k + 1) % weightEvalRes == 0):
                pred = self.pred(w, x)
                if self.cost == "log":
                    costs += [log_loss(y, pred, labels=[0, 1]) + self.regularize(w)]
                elif self.cost == "mse":
                    costs += [mean_squared_error(y, pred, squared=False) / n + self.regularize(w)]
            
        if q != None:
            q.put([w, np.array(costs)])
            
        return w, np.array(costs)
