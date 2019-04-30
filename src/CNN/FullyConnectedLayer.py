import numpy as np
from CNN.AbstractLayer import Layer
import CNN.utils as utils


class FullyConnectedLayer(Layer):

    def __init__(self, weights, learningRate, act_f=utils.sigmoid, isLearning=True):
        super(FullyConnectedLayer, self).__init__(isLearning)
        self._act_f = act_f  # activation function
        self._weights = weights
        self._learningRate = learningRate

    @staticmethod
    def connect(vector, weights, act_f):
        """
        Does the dot product between the input vector and the filter.
        Vector is a   1 x n array
        Filter is a   n x 1 array
        result is a   1 x 1 array for the two nodes of the fully connected layer.
        """
        node = np.dot(vector, weights)
        result = act_f(node)
        return (node, result)

    def squaredError(prediction, actual):
        return 1/2 * np.square(prediction - actual)

    @staticmethod
    def learnFullyConnected(loss, previousLayer, alpha, weights, learningRate, act_f):
        """
        previousLayer : values of the nodes from the previous layer
        alpha : sum(xi * wi) -> used for derivation
        weights : weights vector
        """
        # contains the loss of the previous layer
        previousLayerLoss = np.zeros(previousLayer.shape)
        # will be used to compute the updated weightss
        filtersCorrection = np.zeros(weights.shape)
        print(loss)
        for i in range(weights.shape[0]):  # for i along the height
            df = act_f(alpha, derivative=True)
            filtersCorrection[i] = loss * previousLayer[i] * df
            previousLayerLoss[i] = loss * weights[i] * df

            weights -= learningRate * filtersCorrection

        return previousLayerLoss, weights

    def compute(self, tensor):
        """
        basic computation function, calls the main function
        """
        vector = tensor.flatten()

        node, res = FullyConnectedLayer.connect(vector, self._weights, self._act_f)
        # saves last input and intermediate results
        self.saveData((tensor, node))
        return res

    def learn(self, loss):
        """
        basic learning method, sets some parameters and calls the main function
        """
        previousLayer, alpha = self.getSavedData()
        previousLayerLoss, self._weights = FullyConnectedLayer.learnFullyConnected(loss, previousLayer.flatten(), alpha, self._weights, self._learningRate, self._act_f)
        return previousLayerLoss.reshape(previousLayer.shape)
