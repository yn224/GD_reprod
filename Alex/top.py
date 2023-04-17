#%%
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sag import run_sag
from saga import run_saga
from finito import run_finito

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

for i in range(9):  
  plt.subplot(330 + 1 + i)
  plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
  plt.show()

#--------------------------------------------------------------------
# Main
#--------------------------------------------------------------------
if __name__ == "__main__":
    print("Hello")

