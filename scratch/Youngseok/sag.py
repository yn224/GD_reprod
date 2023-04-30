#====================================================================
# SAG algorithm implementation
#====================================================================
# import numpy as np
from common import *

def run_sag(features, labels, step, iterations):
    n = features.shape[0]
    d = features.shape[1]
    assert n == labels.shape[0]

    # Initialize weight and gradient table
    w = np.zeros(features.shape[1])
    G = np.zeros((features.shape[0], features.shape[1]))

    # Suboptimality gap
    opt_gap = []

    # Step size alpha
    step = 0.001

    # Perform SAG iterations
    for _ in range(iterations):

        # Pick a random i \in {1, ..., n}
        idx = np.random.randint(0, n)
        x = features[idx]
        y = labels[idx]

        # Check if data point has been seen
        if idx not in idxs:
            idxs.append(idx)
            m += 1
        
        # Calculate and store the current gradient
        grad = compute_grad(reg)
        G[idx] = grad

        # Update weight
        g_avg = np.sum(G, axis=0) / m
        w = w - step * g_avg

        # Compute suboptimality gap
        cost = compute_cost(w, features, labels, "logloss")
        opt_gap.append(cost)

    return w, opt_gap
