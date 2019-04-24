from CNN.ConvolutionLayer import ConvolutionLayer
from CNN.ReluLayer import ReluLayer
from CNN.MaxPoolingLayer import MaxPoolingLayer
from CNN.FullyConnectedLayer import FullyConnectedLayer


"""
    Input format :
       3-tensor where a_{ijk} is I don't know what (a float value) for the k'th
       track, the i'th time step and the j'th pitch
"""


class CNN:
    """
        This class implements a CNN, the network is built in buildNetwork.
        TODO : implement backprop and a training function
    """

    def __init__(self, isLearning=False):
        self._layers = []
        self._isLearning = isLearning
        self._output = None
        self.buildNetwork()

    def buildNetwork(self):
        """
            Here is built the network, the layers are stacked.
            First one being the input layer.
            Last one being the output layer.
            The ouput of a layer should be the input of the next.
        """
        raise NotImplementedError

    def predict(self, inputTensor):
        """
            Main "predicting" method of the CNN, based on the learned model,
            it outputs a result (I swear).
        """
        currentTensor = inputTensor
        for layer in self._layers:
            currentTensor = layer.compute(currentTensor)
        if self._isLearning:
            self._output = currentTensor
        return currentTensor

    def train(self, expected):
        """
            Main training Function, it computes the loss of the output layer and
            uses back prop to update the parameters (filters, weights) of the network.
        """
        currentLoss = FullyConnectedLayer.calculateLeastSquares(self._output, expected)
        for i in range(len(self._layers)-1, 0, -1):
            currentLoss = self._layers[i].learn(currentLoss)

    def addConvLayer(self, filters, learningRate, stride=1):
        self._layers.append(ConvolutionLayer(filters,
                                             learningRate,
                                             stride,
                                             self._isLearning))

    def addReluLayer(self):
        self._layers.append(ReluLayer(self._isLearning))

    def addPoolingLayer(self, partitionSize=2):
        self._layers.append(MaxPoolingLayer(partitionSize, self._isLearning))

    def addFullyConnectedLayer(self, filters, learningRate):
        self._layers.append(FullyConnectedLayer(filters, learningRate))
