#====================================================================
# Collection of common operations performed by algorithms
#====================================================================
import math
import numpy as np
from sklearn.metrics import log_loss

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Log-loss objective value
def logloss_cost(w, x, y):
    y_pred = sigmoid(w @ x)
    return log_loss(y, y_pred, labels=[0, 1])

# Log-loss gradient value
def logloss_grad(w, x, y):
    return (sigmoid(w @ x) - y) * x

def lse_cost():
    return 0

def lse_grad():
    return 0

# L1 regularization gradient - soft thresholding
def l1_grad(x_tilde, lmda, alpha):
    la = lmda * alpha
    idx_case1 = np.argwhere(x_tilde >= la).flatten()
    idx_case3 = np.argwhere(x_tilde <= -la).flatten()
    x = np.zeros(x_tilde.shape)
    x[idx_case1] = x_tilde[idx_case1] - la
    x[idx_case3] = x_tilde[idx_case3] + la
    return x

# L2 regularization gradient
def l2_grad(lmda, x):
    return 2 * lmda * x

# L1 regularization cost
def l1_cost(lmda, x):
    return lmda * np.linalg.norm(x, ord=1)

# L2 regularization cost
def l2_cost(lmda, x):
    return lmda * np.linalg.norm(x)

# reg: regularization term: 1 if L1, 2 if L2
def compute_grad(lmda, w, x, y, losstype, reg):
    res = 0

    # Compute respective gradient
    if losstype == "logloss":
        res += logloss_grad(w, x, y)
    elif losstype == "lse":
        res += lse_grad()

    # Compute respective regularization
    if reg == 1:
        res += l1_grad()
    elif reg == 2:
        res += l2_grad(lmda, x)
    
    return res

# Compute the cost function value
def compute_cost(lmda, w, x, y, losstype, reg):
    res = 0

    # Compute respective gradient
    if losstype == "logloss":
        res += logloss_cost(w, x, y)
    elif losstype == "lse":
        res += lse_cost()

    # Compute respective regularization
    if reg == 1:
        res += l1_cost(lmda, x)
    elif reg == 2:
        res += l2_cost(lmda, x)

    return res