#%%
from keras.datasets import mnist
import matplotlib.pyplot as plt
from common import *
from sag import run_sag
from saga import run_saga
from finito import run_finito
from svrg import run_svrg

def compute_accuracy():
    return str(100) + "%"

#--------------------------------------------------------------------
# Main
#--------------------------------------------------------------------
if __name__ == "__main__":
    # Load the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Shapes 
    # X_train: (60000, 28, 28)
    # Y_train: (60000,)
    # X_test:  (10000, 28, 28)
    # Y_test:  (10000,)

    # w, opt_gap = run_sag(train_X, train_y, (), 0.001, 10000, 50)
    # acc = compute_accuracy()

    # w, opt_gap = run_saga(train_X, train_y, (), 0.001, 10000, 50)
    # acc = compute_accuracy()

    # w, opt_gap = run_finito(train_X, train_y, (), 0.001, 10000, 50)
    # acc = compute_accuracy()

    w, opt_gap = run_svrg(train_X, train_y, 0.001, 5, 3)
    # acc = compute_accuracy()
