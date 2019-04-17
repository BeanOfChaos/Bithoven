import numpy as np
from CNN.AbstractLayer import Layer


class ConvolutionLayer(Layer):

    def __init__(self, filters, stride=1, isLearning=True):
        super(ConvolutionLayer, self).__init__(isLearning)
        self._filters = filters
        self._stride = stride

    @staticmethod
    def convolve(tensor, filters, stride):
        """
            Convolution layer. It takes a tensor input or the feature map produced by the previous
            layer and applies its convolution according to its own filters.
            stride : the "sliding step" of the convolution, usually 1
            filters : an array of filters (3 dimensional filters)
        """
        featureMap = np.zeros(tuple(list(tensor.shape[:2])+[filters.shape[0]])) # init the resulting feature map
        tensor = np.pad(tensor, ((0, filters.shape[1] - stride), (0, filters.shape[2] - stride), (0,0)), "constant")
        for f in range(filters.shape[0]): # for each 3-dimensional filter
            for i in range(featureMap.shape[0]): # line i
                for j in range(featureMap.shape[2]): # column j
                    # we compute the result of the dot product between the current receptive field and the current filter (3 dimensional dot product)
                    featureMap[i][j][f] = np.tensordot(tensor[i:i+filters.shape[1], j:j+filters.shape[2], :], filters[f], axes=([0,1,2],[0,1,2]))
        return featureMap

    @staticmethod
    def learnConv(loss, receivedInput, filters):
        # dimensions do not match, 
        filters = filters - learningRate*loss*receivedInput
        return previousLayerLoss, newFilters

    def compute(self, tensor):
        self.saveData(tensor)
        return ConvolutionLayer.convolve(tensor, self._filters, self._stride)

    def learn(self, loss):
        if self.isLearning():
            res, self._filters = ConvolutionLayer.learnConv(loss, self.getSavedData(), self._filters)
            return res

