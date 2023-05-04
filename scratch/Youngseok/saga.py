#====================================================================
# SAGA algorithm implementation
#====================================================================
from common import *

def run_saga(features, labels, step, iterations):
    n = features.shape[0]
    d = features.shape[1]
    assert n == labels.shape[0]

    # Initialize weight and gradient table
    w = np.zeros(d)
    G = np.zeros((n, d))

    opt_gap = []

    # For each iteration:
    for k in range(iterations):

        # Pick a random i \in {1, ..., n}
        idx = np.random.randint(0, n)
        x = features[idx]
        y = labels[idx]
        
        # Compute the average gradient
        avg_g = np.sum(G, axis=0) / n

        # Update weight
        grad = sigmoid(w * x)
        w -= step * (grad - G[idx] + avg_g)

        # Update gradient table
        G[idx] = grad

        # Compute suboptimality gap
        cost = compute_cost(w, features, labels, "logloss")
        opt_gap.append(cost)

    return w, opt_gap
