import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from algorithms import *
from multiprocessing import Process, Queue

def compute_accuracy(w, featr, label):
    pred = featr @ w
    pred = np.where(pred>0.5, 1, pred)
    pred = np.where(pred<=0.5, 0 ,pred)
    return accuracy_score(pred, label)

def cross_validate(model, features, labels, step, iters, lmda, q):
    n = features.shape[0]
    v_size = n // 10
    cv_acc = []
    for i in range(0, n, v_size):
        cpy_feature = features
        cpy_label = labels

        idxs = list(range(i, i+v_size))
        v_feature = features[idxs, :]
        v_label = labels[idxs, :]

        tt_feature = np.delete(cpy_feature, idxs, axis=0)
        tt_label = np.delete(cpy_label, idxs)

        w, _ = model(tt_feature, tt_label, step, iters, L=lmda)

        acc = compute_accuracy(w, v_feature, v_label)
        cv_acc.append(acc)

    if q != None:
        q.put([lmda, cv_acc])

    return cv_acc

X, y = load_svmlight_file("dataset/mnist.scale.bz2")
y = convert_labels(y)
y = y.reshape(len(y),1)
X = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

q = Queue()
processes = []
step = 0.008
iterations = 1000000
model = Logistic_Regression_SAG

lmda = 0.1
while lmda < 0.3:
    processes.append(Process(target=cross_validate, args=(model, X_train, y_train, step, iterations, lmda, q)))
    lmda += 0.01

for p in processes:
    p.start()

for p in processes:
    p.join()

while not q.empty():
    lmda, acc = q.get()
    score = np.sum(acc)
    print(f"For {lmda}, score: {score}")