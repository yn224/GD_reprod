#====================================================================
# SVRG algorithm implementation
#====================================================================
import numpy as np
import matplotlib.pyplot as plt

# Features: n x d (n samples, each x \in R^d)
# Labels: n x 1 (n samples, each y \in R)
# Squared loss \psi_i = (w^T x_i - y_i)^2
def run_svrg(features, labels, test, step, iterations):
    print("run_svrg")
    w = np.zeros(10)
    update_freq = 0

    grad = 0
    prev_grad = 0

    opt_gap = []

    n = 100
    grad_tbl = np.zeros((10, 10))

    # For each iteration:
    for k in range(iterations):
        
        w_tilde = w

        # Keep average gradient
        mu_tilde = np.sum(grad_tbl) / n

        w0 = w_tilde

        for t in range(update_freq):
            i_t = np.random.randint(0, 10)
            w = w - step * ()
        # Compute gradient

        # Update weight

        # Compute cost

        opt_gap.append()
    return w, opt_gap
