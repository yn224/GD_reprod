#%%
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np

from algs import *
from multiprocessing import Process, Queue

def compute_result(datas, models, regs, path):
    for data in datas:
        # Load data and split train and test
        X, y = load_svmlight_file(data)
        # y = convert_labels(y)
        y = y.reshape(len(y),1)
        X = X.toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # Collect costs of different algorithms
        cost_q = Queue()
        processes = []
        iterations = 1000000
        for model in models:
            proc = Process(target=model, args=(X_train, y_train, 0.005, iterations, regs, cost_q))
            processes.append(proc)

        # Run in parallel
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        # Plot the results
        weightEvalRes = 20000
        i = 0
        while not cost_q.empty():
            alg_name = f"model_{i}"
            w, cost = cost_q.get()
            plt.plot(np.arange(len(cost)) * weightEvalRes, cost, label=alg_name, alpha=0.7)
            i += 1

        # Save figures
        plt.xlabel("Iterations")
        plt.ylabel("Log Loss")
        plt.title(f"Loss of {str(data)}")
        plt.legend()
        plt.yscale("log")
        plt.savefig(f"{path}/{str(data)}_loss.png")

def convert_labels(y):
    y = np.where(y<5, 0, y)
    return np.where(y>4, 1, y)

def Logistic_Regression_SGD(x, y, eta, K, L=0, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        #a = eta/np.sqrt(k+1)
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w))

        #Update weights
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        w = w - a*(grad + L*w)

        #Compute cost
        if ((k + 1) % weightEvalRes == 0):
            pred = bound(sigmoid(x@w))
            costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

if __name__ == '__main__':
    # Parse command-line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dataset", nargs='+', choices=["mnist", "ijcnn", "covtype"], help="Dataset to run tests")
    # parser.add_argument("-a", "--algorithm", nargs='+', choices=["sgd", "sag", "finito", "saga"], help="Algorithms to test")
    # parser.add_argument("-l", "--losstype", nargs='+', choices=["log", "mse"], help="Type of loss function to apply")
    # parser.add_argument("-r", "--regularize", type=int, choices=[1, 2], help="Regularization to apply (L1 or L2)")
    # args = parser.parse_args()

    # Prepare to run tests
    # data_to_run = {
    #     "mnist" : "dataset/mnist.scale.bz2",
    #     "ijcnn" : "dataset/ijcnn1.bz2",
    #     "covtype" : "dataset/covtype.libsvm.binary.scale.bz2"
    # }
    # alg_to_run = {
    #     "sgd" : SGD,
    #     "sag" : SAG,
    #     "finito" : Finito,
    #     "saga" : SAGA
    # }
    
    # datas = []
    # if args.dataset:
    #     for d in args.dataset:
    #         datas.append(data_to_run[d])
    # else:
    #     datas = list(data_to_run.values())

    # models = []
    # if args.algorithm:
    #     for a in args.algorithm:
    #         models.append(alg_to_run[a])
    # else:
    #     models = list(alg_to_run.values())

    # if args.losstype:
    #     models = list(map(lambda x: x[args.losstype], models))
    # else:
    #     models = list(map(lambda x: x["log"], models))

    # if args.regularize:
    #     reg = args.regularize
    # else:
    #     reg = 2

    # print(datas, models, reg)

    # Path config for images
    # path = os.path.dirname(os.path.abspath(__file__))
    # if not os.path.exists("images"):
    #     os.makedirs("images")
    # path = os.path.join(path, "images")

    # compute_result(datas, models, reg, path)

    X, y = load_svmlight_file("dataset/mnist.bz2")
    y = convert_labels(y)
    y = y.reshape(len(y),1)
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    iterations = 1000000
    w, b, costs = Logistic_Regression_SGD(X_train, y_train, 0.005, iterations)
    plt.plot(np.arange(len(costs))*weightEvalRes, costs, label="Orig", alpha=0.7)

    model_sgd = SGD()
    w, costs = model_sgd.run(X_train, y_train, 0.005, iterations)
    plt.plot(np.arange(len(costs))*weightEvalRes, costs, label="Mine", alpha=0.7)

    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("MNIST")
    plt.legend()
    plt.yscale("log")
    plt.show()

# %%
