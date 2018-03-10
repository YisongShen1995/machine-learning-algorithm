from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    return np.mean((y_true-y_pred)**2)
    raise NotImplementedError


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    #print(len(real_labels))
    tp=0.0
    fp=0.0
    fn=0.0
    tn=0.0
    for i in range (0,len(real_labels)):
        if real_labels[i]==predicted_labels[i]:
            if predicted_labels[i]>0:
                tp=tp+1.0
                #print(tp)
            else:
                tn=tn+1.0
                #print(tn)
        else:
            if predicted_labels[i]>0:
                fp=fp+1.0
                #print(fp)
            else:
                fn=fn+1.0
                #print(fn)
        #print(i)
    #print(tp)
    #print(fp)

    F1=2*tp/(2*tp+fn+fp)
    return F1
    raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    A=np.array(features)
    #print(A)
    #print(k)
    for i in range (1,k+1):
        #print(i)
        if i==1:
            X=A
        else:
            X = np.c_[X,A**i]
        #print(X)
    return X.astype(float).tolist()
    raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    point1=np.array(point1)
    point2=np.array(point2)
    dis=np.sqrt(np.sum((point1-point2)**2))
    #print(dis)
    return dis.astype(float).tolist()
    raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    point1=np.array(point1)
    point2=np.array(point2)
    dis=np.dot(point1.T,point2)
    return dis.astype(float).tolist()
    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    point1=np.array(point1)
    point2=np.array(point2)
    dis=-np.exp(-np.sum((point1-point2)**2)/2.0)
    return dis.astype(float).tolist()
    raise NotImplementedError

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        for i in range (0,len(features)):
            temp=np.array(features[i])
            num=np.sqrt(np.sum(temp**2))
            for j in range (0,len(features[i])):
                if num!=0:
                    features[i][j]=features[i][j]/num
        return features
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.w = 0
        self.diff = []
        self.mini = []
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if self.w == 0:
            self.w = 1
            temp=np.array(features)
            maxi=temp.max(axis=0)
            self.mini=temp.min(axis=0)
            self.diff=maxi-self.mini
            temp=temp-self.mini
            temp=np.true_divide(temp,self.diff)
        #print(temp)
        elif self.w==1:
            temp=np.array(features)
            temp=np.true_divide(temp-self.mini,self.diff)
        return temp.astype(float).tolist()
        raise NotImplementedError
"""def normalize(features: List[List[float]]) -> List[List[float]]:
    
    normalize the feature vector for each sample . For example,
    if the input features = [[3, 4], [1, -1], [0, 0]],
    the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
    
    for i in range (0,len(features)):
        temp=np.array(features[i])
        num=np.sqrt(np.sum(temp**2))
        for j in range (0,len(features[i])):
            if num!=0:
                features[i][j]=features[i][j]/num
    return features
    raise NotImplementedError


def min_max_scale(features: List[List[float]]) -> List[List[float]]:
    
    normalize the feature vector for each sample . For example,
    if the input features = [[2, -1], [-1, 5], [0, 0]],
    the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
    
    
    temp=np.array(features)
    maxi=temp.max(axis=0)
    mini=temp.min(axis=0)
    diff=maxi-mini
    temp=temp-mini
    temp=np.true_divide(temp,diff)
    print(temp)
    return temp.tolist()
    raise NotImplementedError
"""