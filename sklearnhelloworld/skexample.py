'''
Created on Jul 19, 2017

@author: alysia
'''

from sklearn import random_projection
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def _1_typecasting():
    rng = np.random.RandomState(0)
    X = rng.rand(10,2000)
    X = np.array(X, dtype = 'float32')
    transformer = random_projection.GaussianRandomProjection()
    X_new = transformer.fit_transform(X)
    print(X_new.dtype)
    iris = datasets.load_iris()
    clf = SVC()
    clf.fit(iris.data, iris.target)
    print(clf)

def _2_imageshow():
    digits = datasets.load_digits()
    plt.imshow(digits.images[-1],cmap='Greys')
    
    data1=digits.images[-1].reshape((digits.images[-1].shape[0],-1))
    print(data1)
    plt.show()
    
def _3_knn():
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_Y = iris.target
    
    print(np.unique(iris_Y))
    #print(iris_X)
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    print(iris_Y)
    iris_X_train = iris_X[indices[:-10]]
    iris_Y_train = iris_Y[indices[:-10]]
    iris_X_test = iris_X[indices[-10:]]
    iris_Y_test = iris_Y[indices[-10:]]
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train,iris_Y_train)
    print(knn.predict(iris_X_test))
    print(iris_Y_test)
def _3_1_knn_test():
    train1 = [[1,1],[1,2],[1,3],[0,0],[0,1],[1,5]]
    target1 = [1,1,1,0,0,1]
    knn = KNeighborsClassifier()
    knn.fit(train1[:-1],target1[:-1])
    print(knn)
    print(knn.predict(train1[-1]))
if __name__ == "__main__":
    _3_knn();
    pass