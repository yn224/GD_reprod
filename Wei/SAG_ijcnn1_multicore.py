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
from multiprocessing import Process, Queue
from sklearn.datasets import load_svmlight_file
from algorithms import convert_labels, Predict, Logistic_Regression_Batch_GD, Logistic_Regression_SGD, Logistic_Regression_GA, Logistic_Regression_SAG, Logistic_Regression_SAGA
plt.rcParams['figure.figsize'] = [8, 8]

if __name__ == '__main__':
    #Load data and split train and test
    X, y = load_svmlight_file("ijcnn1.bz2")
    X = X.toarray()
    y = np.where(y==-1, 0, y)
    y = y.reshape(len(y),1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    q5 = Queue()
    
    tic = time.perf_counter()
    iterations = 20000
    p1 = Process(target=Logistic_Regression_Batch_GD, args=(X_train, y_train, 0.0001, iterations, q1))
    p2 = Process(target=Logistic_Regression_SGD, args=(X_train, y_train, 0.0001, iterations, q2))
    p3 = Process(target=Logistic_Regression_GA, args=(X_train, y_train, 0.0001, iterations, q3))
    p4 = Process(target=Logistic_Regression_SAG, args=(X_train, y_train, 0.000825, iterations, 0.01, q4))
    p5 = Process(target=Logistic_Regression_SAGA, args=(X_train, y_train, 0.000825, iterations, 0.01, q5))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    
    w1,b1,costs1 = q1.get()
    w2,b2,costs2 = q2.get()
    w3,b3,costs3 = q3.get()
    w4,b4,costs4 = q4.get()
    w5,b5,costs5 = q5.get()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    
    toc = time.perf_counter()
    print(f"Ran in {toc - tic:0.4f} seconds")
    
    #Test
    y_pred1 = Predict(X_test,w1,b1)
    y_pred2 = Predict(X_test,w2,b2)
    y_pred3 = Predict(X_test,w3,b3)
    y_pred4 = Predict(X_test,w4,b4)
    y_pred5 = Predict(X_test,w5,b5)
    
    #Accuracies
    print("Test Accuracy for batch GD:", accuracy_score(y_test, y_pred1))
    print("Test Accuracy for SGD:", accuracy_score(y_test, y_pred2))
    print("Test Accuracy for GA:", accuracy_score(y_test, y_pred3))
    print("Test Accuracy for SAG:", accuracy_score(y_test, y_pred4))
    print("Test Accuracy for SAGA:", accuracy_score(y_test, y_pred5))
    
    #Plots
    plt.plot(np.arange(len(costs1)),costs1, label="Batch GD")
    plt.plot(np.arange(len(costs2)),costs2, label="SGD", alpha=0.7)
    plt.plot(np.arange(len(costs3)),costs3, label="GA")
    plt.plot(np.arange(len(costs4)),costs4, label="SAG")
    plt.plot(np.arange(len(costs5)),costs5, label="SAGA", alpha=0.6)
    
    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Iterations vs Loss")
    plt.legend()
    plt.yscale("log")
    plt.show()
    
    

    
    
    
    
    
    