#%%
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:31:58 2023

@author: Wei
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from multiprocessing import Process, Queue
from sklearn.datasets import load_svmlight_file
from algorithms import convert_labels, Logistic_Regression_SGD, Logistic_Regression_SAG, Logistic_Regression_SAGA, Logistic_Regression_finito, Predict
plt.rcParams['figure.figsize'] = [8, 8]

if __name__ == '__main__':
    X, y = load_svmlight_file("dataset/mnist.scale.bz2")
    y = convert_labels(y)
    y = y.reshape(len(y),1)
    X = X.toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    
    tic = time.perf_counter()
    iterations = 1000000
    lmda = 0.2
    lmdas = []
    while lmda < 0.3:
        lmdas.append(lmda)
        lmda += 0.01
    processes = [Process(target=Logistic_Regression_SAG, args=(X_train, y_train, 0.008, iterations, l, q2)) for l in lmdas]
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

    weightEvalRes = 20000
    i = 0.2
    while not q2.empty():
        w, cost = q2.get()
        plt.plot(np.arange(len(cost))*weightEvalRes, cost, label=str(i), alpha=0.7)
        i += 0.01
    
    toc = time.perf_counter()

    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("MNIST")
    plt.legend()
    plt.yscale("log")
    plt.show()

# %%
