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
plt.rcParams['figure.figsize'] = [10, 5]

if __name__ == '__main__':
    #Load data and split train and test
    # digits = load_digits()
    # X = digits.data
    # y = convert_labels(digits.target).reshape(len(digits.target),1)

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
    n = X_train.shape[0]
    epochs = 15
    iterations = n*epochs
    L = 0.001
    p1 = Process(target=Logistic_Regression_SGD, args=(X_train, y_train, 0.005, iterations, L, q1, n))
    p2 = Process(target=Logistic_Regression_SAG, args=(X_train, y_train, 0.008, iterations, L, q2, n))
    p3 = Process(target=Logistic_Regression_SAGA, args=(X_train, y_train, 0.008, iterations, L, q3, n))
    p4 = Process(target=Logistic_Regression_finito, args=(X_train, y_train, 450, iterations, 0.05, q4, n))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    w1,b1,costs1 = q1.get()
    w2,b2,costs2 = q2.get()
    w3,b3,costs3 = q3.get()
    w4,b4,costs4 = q4.get()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    
    toc = time.perf_counter()
    print(f"Ran in {toc - tic:0.4f} seconds")

    print("Accuracies:")
    print("SGD:", accuracy_score(y_test, Predict(X_test,w1,0)))
    print("SAG:", accuracy_score(y_test, Predict(X_test,w2,0)))
    print("SAGA:", accuracy_score(y_test, Predict(X_test,w3,0)))
    print("Finito:", accuracy_score(y_test, Predict(X_test,w4,0)))
    
    #Plots
    plt.plot(np.arange(len(costs1)),costs1, label="SGD", alpha=0.7)
    plt.plot(np.arange(len(costs2)),costs2, label="SAG", alpha=0.7)
    plt.plot(np.arange(len(costs3)),costs3, label="SAGA", alpha=0.7)
    plt.plot(np.arange(len(costs4)),costs4, label="Finito", alpha=0.7)

    plt.xlabel("Epochs")
    plt.ylabel("Full gradient norm")
    plt.title("MNIST")
    plt.legend()
    plt.yscale("log")
    plt.show()
    
    

    
    
    
    
    
    