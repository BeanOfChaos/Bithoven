import numpy as np
from CNN.AbstractLayer import Layer


class ConvolutionLayer(Layer):

    def __init__(self, filters, learningRate, stride=1, isLearning=True):
        super(ConvolutionLayer, self).__init__(isLearning)
        self._learningRate = learningRate
        self._filters = filters
        self._stride = stride

    @staticmethod
    def convolve(tensor, filters, stride):
        """
            Convolution layer.
            It takes a tensor input or the feature map produced by the previous
            layer and applies its convolution according to its own filters.
            stride : the "sliding step" of the convolution, usually 1
            filters : an array of filters (3 dimensional filters)
        """
        # init the resulting feature map
        featureMap = np.zeros(tuple(list(tensor.shape[:2])+[filters.shape[0]]))

        print(filters.shape, tensor.shape, featureMap.shape)

        tensor = np.pad(tensor, ((0, filters.shape[1] - stride),
                        (0, filters.shape[2] - stride), (0, 0)), "constant")
        for f in range(filters.shape[0]):  # for each 3-dimensional filter
            for i in range(featureMap.shape[0]):  # line i
                for j in range(featureMap.shape[1]):  # column j
                    # we compute the result of the dot product between:
                    # (1) the current receptive field
                    # and (2) the current filter (3 dimensional dot product)
                    featureMap[i][j][f] \
                            = np.tensordot(tensor[i:i+filters.shape[1],
                                           j:j+filters.shape[2], :],
                                           filters[f],
                                           axes=([0, 1, 2], [0, 1, 2]))
        return featureMap

    @staticmethod
    def learnConv(loss, receivedInput, filters, learningRate):
        """
            Function computing the loss of the previous layer and the updated filters.
            The received loss is computed in the next layer and sent here through backprop.
        """
        # contains the loss of the previous layer
        previousLayerLoss = np.zeros(receivedInput.shape)
        # will be used to compute the updated filters
        filtersCorrection = np.zeros(filters.shape)
        for i in range(filters.shape[0]):  # for each filter
            for j in range(filters.shape[1]):  # for i along the height
                for k in range(filters.shape[2]):  # for j along the width
                    # computing dL/dinput and dL/dW
                    previousLayerLoss[j:j+filters.shape[1],
                                      k:k+filters.shape[2], :] \
                                        += loss[j, k, i] * filters[i]
                    filtersCorrection[i] += loss[j, k, i] \
                        * receivedInput[j:j+filters.shape[1],
                                        k:k+filters.shape[2],
                                        :]
        # returns the previous layer's loss and the updated filters
        return previousLayerLoss, filters - learningRate * filtersCorrection

    def compute(self, tensor):
        """
            Wraps the computation static method
        """
        self.saveData(tensor)
        return ConvolutionLayer.convolve(tensor, self._filters, self._stride)

    def learn(self, loss):
        """
            Wraps the learning static method and update the filters
        """
        if self.isLearning():
            res, self._filters = ConvolutionLayer.learnConv(loss,
                                                            self.getSavedData(),
                                                            self._filters,
                                                            self._learningRate)
            return res
