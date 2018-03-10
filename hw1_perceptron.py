from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        for j in (1,self.max_iteration):
            converge = True
            for i in range (len(features)):
                w = np.array(self.w)
                data = np.array(features[i])
                sumd = np.dot(w.T,data)
                d = np.sqrt(np.sum(w**2))
                k = sumd/(d+0.000000001)#0.000000001 is epsilon to prevent zero

                if k>self.margin:
                    data_sign=1
                    if data_sign==labels[i]:
                        pass
                    else:
                        w = w+labels[i]*data/np.sqrt(np.sum(data**2))
                        self.w = w.astype(float).tolist()
                        converge = False
                elif k<-self.margin:
                    data_sign=-1
                    if data_sign==labels[i]:
                        pass
                    else:
                        w = w+labels[i]*data/np.sqrt(np.sum(data**2))
                        self.w = w.astype(float).tolist()
                        converge = False
                else:
                    w = w+labels[i]*data/np.sqrt(np.sum(data**2))
                    self.w = w.astype(float).tolist()
                    converge = False   
            if converge:
                break
        return converge
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        raise NotImplementedError
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        result = []
        for i in range (len(features)):
            w=np.array(self.w)
            features_array=np.array(features[i])
            data=np.dot(w.T,features_array)
            if data>0:
                result.append(1)
            else:
                result.append(-1)
        return result
            
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        return self.w
    