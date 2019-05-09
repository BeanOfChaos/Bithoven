import numpy as np
from CNN.AbstractLayer import Layer
import CNN.utils as utils


class FullyConnectedLayer(Layer):

    def __init__(self, weights, learningRate, act_f=utils.reLu, isLearning=True):
        super(FullyConnectedLayer, self).__init__(isLearning)
        self._act_f = act_f  # activation function
        self._weights = weights
        self._learningRate = learningRate

    @staticmethod
    def connect(vector, weights, act_f):
        """
        Does the dot product between the input vector and the filter.
        Vector is a   1 x n array
        Filter is a   n x m array
        result is a   1 x m array for the two nodes of the fully connected layer.
        """
        node = np.dot(vector, weights)
        result = act_f(node)
        return (node, result)

    def squaredError(prediction, actual):
        return 1/2 * np.square(prediction - actual)

    @staticmethod
    def learnFullyConnected(loss, previousLayer, alpha, weights, learningRate, act_f):
        """
        previousLayer : input value received at last forward pass
        alpha : output value at last forward pass, before activation
        weights : weights matrix
        """
        df = act_f(alpha, derivative=True)

        weightsCorrection = np.matmul(previousLayer, df * loss)
        previousLayerLoss = np.matmul(weights, df * loss)
        weights -= learningRate * weightsCorrection

        return previousLayerLoss, weights

    def compute(self, tensor):
        """
        basic computation function, calls the main function
        """
        vector = tensor.flatten()
        alpha, res = FullyConnectedLayer.connect(vector, self._weights, self._act_f)
        # saves last input and intermediate results
        self.saveData((tensor, alpha))
        return res

    def learn(self, loss):
        """
        basic learning method, sets some parameters and calls the main function
        """
        previousLayer, alpha = self.getSavedData()
        previousLayerLoss, self._weights = FullyConnectedLayer.learnFullyConnected(loss, previousLayer.flatten(), alpha, self._weights, self._learningRate, self._act_f)
        return previousLayerLoss.reshape(previousLayer.shape)
