import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from algorithms import *
from multiprocessing import Process, Queue
import pandas as pd 
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

# No testing yet!!!

if __name__ == '__main__':
    # Load dataset and split into train and test
    X, y = load_svmlight_file("dataset/ijcnn1.bz2")
    y = y.reshape(len(y),1)
    X = X.toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()

    # to be tuned
    iterations = 600000

    # 0.001 to be tuned
    p1 = Process(target=Linear_Regression_SGD, args=(X_train, y_train, 0.0001, iterations, 0, q1))
    p2 = Process(target=Linear_Regression_SAG, args=(X_train, y_train, 0.0001, iterations, 0, q2))
    p3 = Process(target=Linear_Regression_SAGA, args=(X_train, y_train, 0.0001, iterations, 0, q3))
    p4 = Process(target=Linear_Regression_finito, args=(X_train, y_train, 200, iterations, 0, q4))

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

    y_pred1 = Linear_Predict(X_test,w1)
    y_pred2 = Linear_Predict(X_test,w2)
    y_pred3 = Linear_Predict(X_test,w3)
    y_pred4 = Linear_Predict(X_test,w4)

   
    #Plots
    plt.plot(np.arange(len(costs1)),costs1, label="SGD")
    plt.plot(np.arange(len(costs2)),costs2, label="SAG")
    plt.plot(np.arange(len(costs3)),costs3, label="SAGA")
    plt.plot(np.arange(len(costs4)),costs4, label="Finito")

    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Mean squared error loss of different gradient methods")
    plt.legend()
    plt.yscale("log")
    plt.show()