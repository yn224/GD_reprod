#====================================================================
# Finito algorithm implementation
#====================================================================
from common import *

def run_finito(features, labels, mu, iterations):
    n = features.shape[0]
    d = features.shape[1]
    assert n == labels.shape[0]

    # Initialize weights and tables
    w = np.zeros(n)
    P = np.zeros((n, d)) # Weight table
    G = np.zeros((n, d)) # Gradient table

    opt_gap = []

    # TODO: you can change this to whatever alpha value it is
    # - pg. 7 of Finito paper.
    alpha = 2

    # For each iteration
    for k in range(iterations):        
        # Learning rate - mu is convexity constant
        a = 1/(alpha * mu * k)

        # Draw random sample with replacement
        idx = np.random.randint(0, n)
        x = features[idx]
        y = labels[idx]

        # Take the average of the weights \phi
        phi_bar = np.sum(P, axis=0) / n

        # Take the average of the gradients
        g_bar = np.sum(G, axis=0) / n

        # Update weight
        w = phi_bar - a * g_bar
        b = b - a * (y_pred - y)
        
        # Store data
        P[idx] = w
        G[idx] = grad
        
        # Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        # Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((64,1))

        # Compute suboptimality gap
        cost = compute_cost(w, features, labels, "logloss")
        opt_gap.append(cost)
        
    return w, opt_gap