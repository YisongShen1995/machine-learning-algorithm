from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function
        self.features=[[]]
        self.labels=[]
    def train(self, features: List[List[float]], labels: List[int]):
        self.features=features
        self.labels=labels
       #print(features)
       # raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        labels=[]
        for i in range (0,len(features)):
            dist=[]
            for j in range (0,len(self.features)):
                dist.append(self.distance_function(features[i],self.features[j]))
            sortDs = numpy.argsort(dist)
            countNumber = {}
            for x in range(self.k):
                tempLable=self.labels[sortDs[x]]
                countNumber[tempLable]=countNumber.get(tempLable,0)+1
                #print(countNumber)
                #print(max(countNumber.items(),key=lambda x:x[1])[0])
            labels.append(max(countNumber.items(),key=lambda x:x[1])[0])
        return labels        
        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
