import numpy as np
from CNN.AbstractLayer import Layer

class FullyConnectedLayer(Layer):

    def __init__(self, weights, learningRate, stride=1, isLearning=True):
        super(FullyConnectedLayer, self).__init__(isLearning)
        self._weights = weights
        self._learningRate = learningRate

    @staticmethod
    def sigmoid(value):
        """
        Basic Sigmoid calculation
        """
        print(value)
        #return np.exp(value)/(np.exp(value) + 1)
        print(1/(np.exp(-value) + 1))
        return 1/(np.exp(-value) + 1)

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
        return 1/2 * np.square(prediction-actual)

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
            filtersCorrection[i] = loss * (previousLayer[i] * np.exp(alpha))/np.square((np.exp(alpha)+1))
            previousLayerLoss[i] = loss * (weights[i] * np.exp(alpha))/np.square((np.exp(alpha)+1))

            weights = weights - learningRate*filtersCorrection

        return previousLayerLoss, weights


    def compute(self, tensor):
        """
        basic computation function, calls the main function
        """
        vector = np.reshape(tensor, -1)
        res = FullyConnectedLayer.connect(vector, self._weights)
        self.saveData((tensor, res))
        return res

    def learn(self, loss):
        """
        basic learning method, sets some parameters and calls the main function
        """
        (previousLayer, alpha) = self.getSavedData()
        previousLayerLoss, self._weights = FullyConnectedLayer.learnFullyConnected(loss, previousLayer.reshape(-1), alpha, self._weights, self._learningRate)

        return previousLayerLoss.reshape(previousLayer.shape)
