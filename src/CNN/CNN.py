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

    def __init__(self, isLearning=False, trainingSet=None):
        self._layers = []
        self._isLearning = isLearning
        self._trainingSet = trainingSet
        self.buildNetwork()
        if isLearning:
            self.train()

    def buildNetwork(self):
        """
            Here is built the network, the layers are stacked, the first one being the input layer,
            the last one being the output layer. The ouput of a layer is the input of the next
        """
        pass

    def predict(self, inputTensor):
        """
            Main "predicting" method of the CNN, based on what is learned and the architecture of
            the network, it outputs a result
        """
        currentTensor = inputTensor
        for layer in self._layers:
            currentTensor = layer.compute(currentTensor)
        return currentTensor

    def train(self):
        """
            Main training Function, it computes the loss of the output layer and
            uses back prop to update the parameters (filters, weights) of the network
        """
        pass

    def addConvLayer(self, filters, learningRate, stride=1):
        self._layers.append(ConvolutionLayer(filters, learningRate, stride, self._isLearning))

    def addReluLayer(self):
        self._layers.append(ReluLayer(self._isLearning))

    def addPoolingLayer(self, partitionSize=2):
        self._layers.append(MaxPoolingLayer(partitionSize, self._isLearning))

    def addFullyConnectedLayer(self, filters):
        self._layers.append(FullyConnectedLayer(filters))
