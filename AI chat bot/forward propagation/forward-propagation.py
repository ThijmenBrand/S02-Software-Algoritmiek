from multiprocessing.dummy import Array
import numpy as np
from pandas import array

class Neural_Network(object):
    def __init__(self):        
        #Defineer het netwerk zijn vorm
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #geef het netwerk random weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

NN = Neural_Network()
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
yHat = NN.forward(X)
print(yHat)