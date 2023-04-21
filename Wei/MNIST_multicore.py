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
from algorithms import convert_labels, Predict, Logistic_Regression_Batch_GD, Logistic_Regression_SGD, Logistic_Regression_GA, Logistic_Regression_SAG, Logistic_Regression_SAGA
plt.rcParams['figure.figsize'] = [8, 8]

if __name__ == '__main__':
    #Load data and split train and test
    digits = load_digits()
    X = digits.data
    y = convert_labels(digits.target).reshape(len(digits.target),1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    
    tic = time.perf_counter()
    iterations = 16000
    p1 = Process(target=Logistic_Regression_SGD, args=(X_train, y_train, 0.001, iterations, q1))
    p2 = Process(target=Logistic_Regression_SAG, args=(X_train, y_train, 0.000675, iterations, q2))
    p3 = Process(target=Logistic_Regression_SAGA, args=(X_train, y_train, 0.000675, iterations, q3))
    
    p1.start()
    p2.start()
    p3.start()
    
    w1,b1,costs1 = q1.get()
    w2,b2,costs2 = q2.get()
    w3,b3,costs3 = q3.get()
    
    p1.join()
    p2.join()
    p3.join()
    
    toc = time.perf_counter()
    print(f"Ran in {toc - tic:0.4f} seconds")
    
    #Plots
    plt.plot(np.arange(len(costs1)),costs1, label="SGD", alpha=0.7)
    plt.plot(np.arange(len(costs2)),costs2, label="SAG", alpha=0.7)
    plt.plot(np.arange(len(costs3)),costs3, label="SAGA", alpha=0.7)

    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Iterations vs Loss")
    plt.legend()
    plt.yscale("log")
    plt.show()
    
    

    
    
    
    
    
    