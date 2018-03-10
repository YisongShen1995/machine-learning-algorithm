from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.theta = []
    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        A = numpy.array(features)
        X = numpy.insert(A,0,1,axis=1)
        #print(numpy.dot(X.T,X))
        #print(X)
        
        X_ = numpy.linalg.pinv(numpy.dot(X.T,X))#I use pinv instead inv here because under vmware ubuntu 16.04 python3.5.2 when k get large the result is not correct
        y = numpy.array(values)
        #print(y)
        self.theta =numpy.dot(numpy.dot(X_,X.T),y)
        #print(self.theta)

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        A = numpy.array(features)
        X = numpy.insert(A,0,1,axis=1)
        y = numpy.dot(X,self.theta)
        return y.astype(float).tolist()
    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.theta.astype(float).tolist()


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.theta = []
    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        A = numpy.array(features)
        X = numpy.insert(A,0,1,axis=1)
        B = numpy.dot(X.T,X)
        C = self.alpha*numpy.eye(B.shape[0])
        X_ = numpy.linalg.pinv(B+C)
        y = numpy.array(values)
        self.theta = numpy.dot(numpy.dot(X_,X.T),y)

    def predict(self, features: List[List[float]]) -> List[float]:
        A = numpy.array(features)
        X = numpy.insert(A,0,1,axis=1)
        y = numpy.dot(X,self.theta)
        return y.astype(float).tolist()
    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.theta.astype(float).tolist()


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
