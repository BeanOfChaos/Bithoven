import numpy as np
from CNN.AbstractLayer import Layer

class FullyConnectedLayer(Layer):

    def __init__(self, weights, learningRate, isLearning=True):
        super(FullyConnectedLayer, self).__init__(isLearning)
        self._weights = weights
        self._learningRate = learningRate

    @staticmethod
    def sigmoid(value):
        """
        Basic Sigmoid calculation
        """
        #return np.exp(value)/(np.exp(value) + 1)
        return 1/(np.exp(-value) + 1)

    @staticmethod
    def connect(vector, weights):
        """
        Does the dot product between the input vector and the filter.
        Vector is a   1 x n array
        Filter is a   n x 1 array
        result is a   1 x 1 array for the two nodes of the fully connected layer.
        """
        node = np.dot(vector, weights)
        result = FullyConnectedLayer.sigmoid(node)
        return (node, result)

    def squaredError(prediction, actual):
        return 1/2 * np.square(prediction - actual)

    @staticmethod
    def learnFullyConnected(loss, previousLayer, alpha, weights, learningRate):
        """
        previousLayer : values of the nodes from the previous layer
        alpha : sum(xi * wi) -> used for derivation
        weights : weights vector
        """
        # contains the loss of the previous layer
        previousLayerLoss = np.zeros(previousLayer.shape)
        # will be used to compute the updated weightss
        filtersCorrection = np.zeros(weights.shape)
        for i in range(weights.shape[0]):  # for i along the height
            filtersCorrection[i] = loss * previousLayer[i] * np.exp(alpha) / np.square(np.exp(alpha)+1)
            previousLayerLoss[i] = loss * weights[i] * np.exp(alpha) / np.square(np.exp(alpha)+1)

            weights = weights - learningRate * filtersCorrection

        return previousLayerLoss, weights

    def compute(self, tensor):
        """
        basic computation function, calls the main function
        """
        vector = np.reshape(tensor, -1)
        node, res = FullyConnectedLayer.connect(vector, self._weights)
        self.saveData((tensor, node))
        return res

    def learn(self, loss):
        """
        basic learning method, sets some parameters and calls the main function
        """
        previousLayer, alpha = self.getSavedData()
        previousLayerLoss, self._weights = FullyConnectedLayer.learnFullyConnected(loss, previousLayer.reshape(-1), alpha, self._weights, self._learningRate)
        return previousLayerLoss.reshape(previousLayer.shape)

		
