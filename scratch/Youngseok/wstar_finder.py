import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

def convert_labels(y):
    y = np.where(y<5, 0, y)
    return np.where(y>4, 1, y)

def logit(v):
    for i in range(len(v)):
        try:
            res = -1 * math.log((1 / (v[i] + 1e-8)) - 1)
        except:
            res = 0
        v[i] = res
    return v

def compute_wstar(x, y):
    return np.linalg.pinv(x) @ logit(np.linalg.pinv(x.T) @ x.T @ y)

# MNIST
X, y = load_svmlight_file("mnist.scale.bz2")
y = convert_labels(y)
y = y.reshape(len(y),1)
X = X.toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

w_star = compute_wstar(X_train, y_train)

np.savetxt('mnist_wstar.txt', w_star)

# IJCNN1
X, y = load_svmlight_file("ijcnn1.bz2")
X = X.toarray()
y = np.where(y==-1, 0, y)
y = y.reshape(len(y),1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

w_star = compute_wstar(X_train, y_train)

np.savetxt('ijcnn_wstar.txt', w_star)

# Slice
df = pd.read_csv("slice_localization_data.csv")
y = np.array(df["reference"])
y = y.reshape((len(y), 1))
df2 = df.drop("reference", axis=1)
X = np.array(df2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

w_star = compute_wstar(X_train, y_train)

np.savetxt('slice_wstar.txt', w_star)

# COVTYPE
X, y = load_svmlight_file("covtype.libsvm.binary.scale.bz2")
y -= 1
y = y.reshape(len(y),1)
X = X.toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

w_star = compute_wstar(X_train, y_train)

np.savetxt('covtype_wstar.txt', w_star)

# MillionSong
filename = 'YearPredictionMSD.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
y = data[:,0]
y = y.reshape((len(y), 1))
X = data[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

w_star = compute_wstar(X_train, y_train)

np.savetxt('million_wstar.txt', w_star)
