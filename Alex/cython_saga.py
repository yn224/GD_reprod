#====================================================================
# SAGA algorithm implementation in Cython
#====================================================================
# RUN IN JUPYTER NOTEBOOK
%load_ext Cython

# Standard python import
import numpy as np

# Compile-time import of numpy
cimport numpy as np

# Necessary import to use any of use any of numpy PyArray_* API (e.g., shape)
np.import_array()

# Fix the data type
DTYPE = np.float64

def cython_saga(np.ndarray x, np.ndarray y, np.float64 step, np.int n_iters):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    g = np.zeros((x.shape[0], x.shape[1],1)) #Gradient table
    G = np.zeros_like(w) #Gradient table sum
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w))

        #Check if data point has been seen
        if idx not in idxs:
            idxs += [idx]
            m += 1
        
        #Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        #Update weights
        w = w - a*((grad - g[idx] + G/m) + L*w)
        #Update gradient table
        G = G - g[idx] + grad
        #Update previous sample gradient
        g[idx] = grad

        #Compute cost
        if ((k+1)%weightEvalRes==0):
            pred = bound(sigmoid(x@w))
            costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)