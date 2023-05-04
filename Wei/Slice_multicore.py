# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:31:58 2023

@author: Wei
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from multiprocessing import Process, Queue
import pandas as pd
from algorithms import Linear_Regression_SGD, Linear_Regression_SAG, Linear_Regression_SAGA, Linear_Regression_finito, Linear_Predict
plt.rcParams['figure.figsize'] = [10, 5]

if __name__ == '__main__':
    #Load data and split train and test
    df = pd.read_csv("dataset/slice_localization_data.csv")
    y = np.array(df["reference"])
    y = y.reshape((len(y), 1))
    df2 = df.drop("reference", axis=1)
    X = np.array(df2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()

    tic = time.perf_counter()
    n = X_train.shape[0]
    epochs = 15
    iterations = n*epochs
    L = 0.0001
    p1 = Process(target=Linear_Regression_SGD, args=(X_train, y_train, 1e-5, iterations, L, q1, n))
    p2 = Process(target=Linear_Regression_SAG, args=(X_train, y_train, 5e-5, iterations, L, q2, n))
    p3 = Process(target=Linear_Regression_SAGA, args=(X_train, y_train, 1e-4, iterations, L, q3, n))
    p4 = Process(target=Linear_Regression_finito, args=(X_train, y_train, 3.5, iterations, 0.9, q4, n))
    
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
    print("SGD:", r2_score(y_test, Linear_Predict(X_test,w1)))
    print("SAG:", r2_score(y_test, Linear_Predict(X_test,w2)))
    print("SAGA:", r2_score(y_test, Linear_Predict(X_test,w3)))
    print("Finito:", r2_score(y_test, Linear_Predict(X_test,w4)))
    
    #Plots
    plt.plot(np.arange(len(costs1)),costs1, label="SGD", alpha=0.7)
    plt.plot(np.arange(len(costs2)),costs2, label="SAG", alpha=0.7)
    plt.plot(np.arange(len(costs3)),costs3, label="SAGA", alpha=0.7)
    plt.plot(np.arange(len(costs4)),costs4, label="Finito", alpha=0.7)
    
    plt.xlabel("Epochs")
    plt.ylabel("Full gradient norm")
    plt.title("CT Slice")
    plt.legend()
    plt.yscale("log")
    plt.show()
    
    

    
    
    
    
    
    