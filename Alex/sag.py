#====================================================================
# SAG algorithm implementation
#====================================================================
# import numpy as np
from common import *

def run_sag(features, labels, test, iterations, reg=0):
    # Initialize weight and gradient table
    w = np.zeros(features.shape[1])
    G = np.zeros((features.shape[0], features.shape[1]))

    idxs = []

    m = 0

    # Suboptimality gap
    opt_gap = []

    # Step size alpha
    step = 0.001

    # Perform SAG iterations
    for _ in range(iterations):

        # Pick a random i \in {1, ..., n}
        idx = np.random.randint(0, 10)
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

        # Compute and store cost
        gap = compute_cost(reg)
        opt_gap.append(gap)

    return w, np.array(opt_gap)
