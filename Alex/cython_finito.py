#====================================================================
# Finito algorithm implementation in Cython
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

def cython_finito(np.ndarray x, np.ndarray y, np.float64 step, np.int n_iters):
    cdef np.ndarray w = np.zeros([x.shape[1],1], dtype=DTYPE)
    cdef np.ndarray P = np.zeros((x.shape[0], x.shape[1], 1), dtype=DTYPE) # Weight table
    cdef np.ndarray G = np.zeros((x.shape[0], x.shape[1], 1)) # Gradient table
    cdef np.ndarray idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    # TODO: you can change this to whatever alpha value it is
    # - pg. 7 of Finito paper.
    alpha = 2

    # For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):        
        # Learning rate - mu is convexity constant
        a = 1/(alpha * mu)

        # Draw random sample with replacement
        idx = np.random.randint(0,len(x))
        xx = x[idx]
        yy = y[idx]

        # Check if data point has been seen
        if idx not in idxs:
            idxs += [idx]
            m += 1
        
        # Take the average of the weights \phi
        phi_bar = np.sum(P, axis=0) / m

        # Take the average of the gradients
        g_bar = np.sum(G, axis=0) / m

        # Update weight
        w = phi_bar - a * g_bar
        b = b - a * (y_pred-yy)
        
        # Store data
        P[idx] = w
        G[idx] = grad
        
        # Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        # Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((64,1))

        #Compute cost
        pred = bound(sigmoid(x@w+b))
        costs += [log_loss(y, pred, labels = [0,1])]

    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)