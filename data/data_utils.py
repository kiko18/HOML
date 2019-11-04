import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

'''
Data:
    - planar
    - sign
    - happy_house
    - cat_dog
    - cifar10
    - cifar100
'''


def plot_decision_boundary(model, X, y, title):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title(title)
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
    plt.show()
    

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
          
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
    classes = np.array([0, 1])
    return X_train.T, X_test.T, y_train.T, y_test.T, classes



def load_sign_dataset():
    data_ = loadmat('./sign.mat')
    X = data_['X']
    y = data_['y']
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)
    classes = np.array([0, 1, 2, 3, 4, 5])
    return X_train, X_test, y_train.T, y_test.T, classes



def load_cat_dataset():
    data_ = loadmat('./catDog32.mat')
    X = data_['X']
    y = data_['y']
    ratio = 1
    nbOfSample = int(np.round(X.shape[0] / ratio));
    X = X[:nbOfSample, :]
    y = y[:nbOfSample, :]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.16, random_state=42)
    classes = np.array([0, 1])
    return X_train, X_test, y_train.T, y_test.T, classes


def load_happy_house_dataset():
    data = loadmat('./happy_house.mat')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test = data['y_test']
    classes = np.arange(2) #happy or unhappy
    return X_train, X_test, y_train, y_test, classes


def load_cifar10_dataset():
    dataReduction = True
    data = loadmat('./cifar10.mat')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
    
    if dataReduction:
        ratio = 1
        nbOfSample = int(np.round(X_train.shape[0] / ratio));
        X_train = X_train[:nbOfSample, :]
        y_train = y_train[:nbOfSample, :]
        X_test  = X_test[:nbOfSample, :]
        y_test  = y_test[:nbOfSample, :]
      
    classes = np.arange(10)
    return X_train, X_test, y_train.T, y_test.T, classes


def load_cifar100_dataset():
    data = loadmat('./cifar100.mat')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test = data['y_test']
    classes = np.arange(100)
    return X_train, X_test, y_train.T, y_test.T, classes

def load_mnist_dataset():
    data = loadmat('data/mnist.mat')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test = data['y_test']
    classes = np.arange(10)
    return X_train, X_test, y_train.T, y_test.T, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y