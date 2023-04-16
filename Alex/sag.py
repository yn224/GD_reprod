#====================================================================
# SAG algorithm implementation
# Helps from:
#   https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/stochastic_optimization_methods.pdf
#====================================================================
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# SAG
#--------------------------------------------------------------------
def run_sag(features, labels, test, step, iterations):
    w = np.zeros()

    # For each iteration:
    for k in range(iterations):

        # Pick a random i \in {1, ..., n}
        idx = np.random.randint(0, 10)
        
        # Update weight
    return
