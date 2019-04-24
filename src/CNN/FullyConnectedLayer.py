from math import exp, pow
import numpy as np
from AbstractLayer import Layer

class FullyConnectedLayer(Layer):
    
    def __init__(self, weights, learningRate, stride=1, isLearning=True):
        super(FullyConnectedLayer, self).__init__(isLearning)
        self._weights = weights
        self._stride = stride
        self._learningRate = learningRate
        self._predicted = 0
        self._actual = 0
        self._isLearning = isLearning
    
    def sigmoid(value):
        """
        Basic Sigmoid calculation
        """
        return exp(value)/(exp(value) + 1)
    
        
    @staticmethod
    def connect(vector, weights):
        """
        Does the dot product between the input vector and the filter.
        Vector is a   1 x n array
        Filter is a   n x 1 array
        result is a   1 x 1 array for the two nodes of the fully connected layer, on which SoftMax is applied
        """
        node = np.dot(vector, weights)
        result = FullyConnectedLayer.sigmoid(node)
        return result
        

    def calculateLeastSquares(prediction, actual):
        return 1/2 * pow(prediction-actual, 2)
        
        
    @staticmethod
    def learnFullyConnected(loss, previousLayer, alpha, weights, learningRate):
        """
        previousLayer : values of the nodes from the previous layer
        alpha : sum(xi * wi) -> used for derivation
        weights : weights vector
        """

        previousLayerLoss = np.zeros(previousLayer.shape) # contains the loss of the previous layer
        filtersCorrection = np.zeros(weights.shape) # will be used to compute the updated weightss

        for i in range(weights.shape[0]):  #for i along the height
            filtersCorrection[i] = loss * (previousLayer[i] * exp(alpha))/pow((exp(alpha)+1), 2) 
            previousLayerLoss[i] = loss * (weights[i] * exp(alpha))/pow((exp(alpha)+1),2)
        
            weights = weights - learningRate*filtersCorrection
        
        return previousLayerLoss, weights


    def compute(self, tensor):
        """
		basic computation function, calls the main function
        """
        vector = np.reshape(tensor, -1)
        res = FullyConnectedLayer.connect(vector, self._weights)
        self.saveData((vector, res))
        return res

    def learn(self, loss):
        """
        basic learning method, sets some parameters and calls the main function
        """
        (previousLayer, alpha) = self.getData()
        loss = self.calculateLeastSquaresVector(self._prediction, self._actual)
        previousLayerLoss, waights = FullyConnectedLayer.learnFullyConnected(loss, previousLayer, alpha, self._weights, self._learningRate)

        return previousLayerLoss

