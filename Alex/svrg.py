#====================================================================
# SVRG algorithm implementation
#====================================================================
from common import *

def run_svrg(features, labels, step, iterations, inner_iter):
    n = features.shape[0] # 60000
    d = features.shape[1] # 28
    assert n == labels.shape[0]

    # Initialize weight
    w = np.zeros(d)

    # Optimality gap
    opt_gap = []

    # For each iteration:
    for k in range(iterations):

        # Compute the batch gradient
        batch_g = 0
        for i in range(3):
            x = features[i] # 28 x 28
            y = labels[i] # scalar
            batch_g += sigmoid(w)
        batch_g /= n

        # Inner-update of loop
        phi = w
        for _ in range(inner_iter):
            i_t = np.random.randint(0, 10)
            x = features[i_t]
            y = labels[i_t]
            phi -= step * (sigmoid(x @ phi) - sigmoid(x @ w) + batch_g)

        # Update weight
        w = phi

        # Compute suboptimality gap
        cost = compute_cost(w, features, labels, "logloss")
        opt_gap.append(cost)

    return w, opt_gap
